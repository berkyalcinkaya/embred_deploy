import os
import cv2
import argparse
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from typing import List, Union
from .rcnn import ExtractEmbFrame, extract_emb_frame_2d
from .utils import load_model, get_device, class_mapping
from embpred_deploy.models.mapping import model_mapping
from embpred_deploy.config import MODELS_DIR
from .post_process import monotonic_decoding
from tqdm import tqdm

available_models = list(model_mapping.keys())

class_mapping = {0: "t1", 1: "tPN", 2: "tPNf", 3: "t2", 4: "t3", 
                5: "t4", 6: "t5", 7: "t6", 8: "t7", 9: "t8", 10: "tM", 11: "tB", 12: "tEB", 13: "tEmpty"}

SIZE = (224, 224)
NCLASS = 14
RCNN_PATH = os.path.join(MODELS_DIR, "rcnn.pt")

def load_faster_RCNN_model_device(RCNN_PATH, use_GPU=True):
    if use_GPU:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
        
    model = torch.load(RCNN_PATH, map_location=device, weights_only=False)
    model.eval()  # Set to inference mode
    return model, device


def inference(model, device, 
            depths_ims: Union[List[np.ndarray], torch.Tensor, np.ndarray], 
              map_output=True, 
              output_to_str=False, 
              totensor=True, resize=True, 
              normalize=True, 
              get_bbox=True, 
              resnet_normalize=True,
              rcnn_model=None, 
              size=(224, 224),
              ):
    """


    Perform inference on an image using a PyTorch model.
    See documentation for full parameter description.
    """

    assert len(depths_ims) == 3 or depths_ims.shape[-1] == 3, "depths_ims must contain three images."
    if get_bbox:
        assert rcnn_model is not None, "rcnn_model must be provided if get_bbox is True."
        assert totensor, "Image must be converted to a tensor if get_bbox is True."
        assert resize, "Image must be resized if get_bbox is True."
        depths_ims = list(ExtractEmbFrame(depths_ims[0], depths_ims[1], depths_ims[2], rcnn_model, device))

    if isinstance(depths_ims, List):
        image = np.stack(depths_ims, axis=-1)
    else:
        image = depths_ims
    if totensor:
        image = torch.from_numpy(image).permute(2, 0, 1).float()
    if resize:
        image = v2.Resize(size)(image)
    if normalize:
        image /= 255.0
    if resnet_normalize:
        image =v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    
    # Perform inference
    image = image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image).squeeze(0).cpu().numpy()
    
    if map_output or output_to_str:
        output = np.argmax(output).item()
        if output_to_str:
            output = class_mapping[int(output)]
    return output


def main():
    """Main entry point for the embpred_deploy command-line interface."""
    parser = argparse.ArgumentParser(description="Run inference on image(s) with specified focal depths.")
    
    # Mutually exclusive modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single-image",
        type=str,
        help="Path to a single image. The image will be duplicated as an RGB image across all channels."
    )
    group.add_argument(
        "--timelapse-dir",
        type=str,
        default=None,
        help="Path to a directory of images (each image is a timepoint) or a directory containing exactly 3 subdirectories, each with images of a focal depth."
    )
    group.add_argument(
        "--F_neg15",
        type=str,
        help="Path to the F-15 focal depth image."
    )
    group.add_argument(
        "--focal-depths",
        action="store_true",
        help="If provided, indicates that separate focal depth directories are used instead of --timelapse-dir."
    )
    
    # Additional arguments for focal images (used when --F_neg15 is provided)
    parser.add_argument(
        "--F0",
        type=str,
        help="Path to the F0 focal depth image."
    )
    parser.add_argument(
        "--F15",
        type=str,
        help="Path to the F15 focal depth image."
    )
    
    # Focal directories for timelapse images; these are used if --focal-depths is provided.
    parser.add_argument(
        "--focal-dir1",
        type=str,
        default=None,
        help="Path to the first focal depth subdirectory. Required if --focal-depths is provided."
    )
    parser.add_argument(
        "--focal-dir2",
        type=str,
        default=None,
        help="Path to the second focal depth subdirectory. Required if --focal-depths is provided."
    )
    parser.add_argument(
        "--focal-dir3",
        type=str,
        default=None,
        help="Path to the third focal depth subdirectory. Required if --focal-depths is provided."
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=available_models,
        help=f"Name of the model to load from available models: {available_models}"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to the model file. If not provided, the path will be constructed from --model-name."
    )
    
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="If provided, postprocess the model output (map raw output to class labels).",
        default=False
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0'). If not provided, will use CUDA if available, otherwise CPU."
    )

    parser.add_argument(
        "--nth-timepoint",
        type=int,
        default=None,
        help= "If provided, only process the nth timepoint of the timeseries."
    )
    
    args = parser.parse_args()
    
    # If focal-depths flag is set, ensure that all three focal directory arguments are provided.
    if args.focal_depths:
        if not (args.focal_dir1 and args.focal_dir2 and args.focal_dir3):
            parser.error("When using --focal-depths, you must provide --focal-dir1, --focal-dir2, and --focal-dir3.")
    
    # Set device: use provided device or fall back to get_device()
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # Load the models
    model_class = model_mapping[args.model_name][0]
    model_class_arg = model_mapping[args.model_name][1]  # (not used in load_model below)
    model_path = args.model_path if args.model_path is not None else os.path.join(MODELS_DIR, f"{args.model_name}.pth")
    # Use the same device for RCNN model
    use_gpu = device.type == 'cuda'
    rcnn_model, rcnn_device = load_faster_RCNN_model_device(RCNN_PATH, use_GPU=use_gpu)
    model, epoch, best_val_auc = load_model(model_path, device, NCLASS, model_class=model_class, class_args=model_class_arg)
    
    outputs = []
    
    # Timelapse / Focal depths inference branch
    if args.focal_depths:
        # Use explicit focal depth directories (ignores timelapse-dir)
        focal_paths = [args.focal_dir1, args.focal_dir2, args.focal_dir3]
        for path in focal_paths:
            if not os.path.isdir(path):
                print(f"Error: {path} is not a valid directory.")
                exit(1)
        list_files = [sorted(os.listdir(fp)) for fp in focal_paths]
        num_timepoints = min(len(files) for files in list_files)
        for i in tqdm(range(num_timepoints)):
            file_paths = [os.path.join(focal_paths[j], list_files[j][i]) for j in range(3)]
            images = []
            for fp in file_paths:
                img = cv2.imread(fp)
                if img is None:
                    print(f"Failed to load image at {fp}")
                    exit(1)
                images.append(img)
            outputs.append(inference(model, device, images, map_output=False, output_to_str=False,
                                     rcnn_model=rcnn_model))
    elif args.timelapse_dir:
        timelapse_dir = args.timelapse_dir
        if not os.path.isdir(timelapse_dir):
            print(f"Error: {timelapse_dir} is not a valid directory.")
            exit(1)
        subdirs = sorted([d for d in os.listdir(timelapse_dir) if os.path.isdir(os.path.join(timelapse_dir, d))])
        print(f"Subdirectories: {subdirs}")
        if len(subdirs) > 1:
            assert len(subdirs) >= 3, "At least 3 subdirectories are required for timelapse inference."

            if len(subdirs) != 3:
                subdirs = [ "F-15", "F0", "F15" ]
                
            focal_paths = [os.path.join(timelapse_dir, d) for d in subdirs]
            list_files = [sorted(os.listdir(fp)) for fp in focal_paths]
            num_timepoints = min(len(files) for files in list_files)
            for i in tqdm(range(num_timepoints)):

                if args.nth_timepoint is not None and i % args.nth_timepoint != 0:
                    continue

                file_paths = [os.path.join(focal_paths[j], list_files[j][i]) for j in range(3)]
                images = []
                for fp in file_paths:
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Failed to load image at {fp}")
                        exit(1)
                    images.append(img)
                outputs.append(inference(model, device, images, map_output=False, output_to_str=False,
                                         rcnn_model=rcnn_model))
        else:
            # No subdirectories: assume timelapse_dir contains images to be duplicated.
            timepoint_files = sorted([f for f in os.listdir(timelapse_dir) if os.path.isfile(os.path.join(timelapse_dir, f))])
            for file in tqdm(timepoint_files):
                fp = os.path.join(timelapse_dir, file)
                single_image = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if single_image is None:
                    print(f"Failed to load image at {fp}")
                    continue
                duplicated_image = cv2.cvtColor(single_image, cv2.COLOR_GRAY2RGB)
                outputs.append(inference(model, device, duplicated_image.transpose(2, 0, 1),
                                         rcnn_model=rcnn_model, output_to_str=False, map_output=False))
        np.save("raw_timelapse_outputs.npy", np.array(outputs))
        print("Raw outputs saved to raw_timelapse_outputs.npy")
        if args.postprocess:
            outputs = monotonic_decoding(np.array(outputs), loss='NLL')
            print("Postprocessed outputs saved to postprocessed_timelapse_outputs.npy")
        max_prob_classes = np.argmax(np.array(outputs), axis=1)
        print(max_prob_classes)
        np.savetxt("max_prob_classes.csv", max_prob_classes, delimiter=",")
        print("Max probability classes saved to max_prob_classes.csv")
    
        plt.plot(max_prob_classes)
        plt.xlabel("Timepoints")
        plt.ylabel("Max Probability Class")
        plt.title("Max Probability Class Over Time")
        plt.show()
        plt.savefig("max_prob_classes.png")
        print("Plot saved to max_prob_classes.png")
    
    elif args.single_image:
        single_image = cv2.imread(args.single_image, cv2.IMREAD_GRAYSCALE)
        if single_image is None:
            print(f"Failed to load image at {args.single_image}")
            exit(1)
        duplicated_image = cv2.cvtColor(single_image, cv2.COLOR_GRAY2RGB)
        output = inference(model, device, duplicated_image.transpose(2, 0, 1),
                           rcnn_model=rcnn_model, output_to_str=True)
        print(f"Class label: {output}")
    else:
        # Three separate images inference branch
        if not all([args.F_neg15, args.F0, args.F15]):
            print("Error: When not using --single-image or a timelapse mode, all three focal depth images (--F_neg15, --F0, --F15) must be provided.")
            exit(1)
        image_F_neg15 = cv2.imread(args.F_neg15)
        image_F0 = cv2.imread(args.F0)
        image_F15 = cv2.imread(args.F15)
        if image_F_neg15 is None or image_F0 is None or image_F15 is None:
            print("Error: Failed to load one or more focal depth images.")
            exit(1)
        depths_ims = [image_F_neg15, image_F0, image_F15]
        output = inference(model, device, depths_ims,
                           rcnn_model=rcnn_model, output_to_str=True)
        print(f"Class label: {output}")


if __name__ == "__main__":
    main()