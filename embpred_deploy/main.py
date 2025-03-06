import os
import cv2
import argparse
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Union
from .rcnn import ExtractEmbFrame, extract_emb_frame_2d
from .utils import mapping, load_model, get_device
from embpred_deploy.models.mapping import mapping
from embpred_deploy.config import MODELS_DIR
from .post_process import monotonic_decoding
from tqdm import tqdm

available_models = list(mapping.keys())

SIZE = (224, 224)
NCLASS = 13
RCNN_PATH = os.path.join(MODELS_DIR, "rcnn.pt")

def load_faster_RCNN_model_device(RCNN_PATH, use_GPU=True):
    if use_GPU:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
        
    model = torch.load(RCNN_PATH, map_location=device, weights_only=False)
    return model, device


def inference(model, device, depths_ims: Union[List[np.ndarray], torch.Tensor, np.ndarray], 
              map_output=True, output_to_str=False, totensor=True, resize=True, normalize=True, get_bbox=True, 
              rcnn_model=None, size=(224, 224)):
    """
    Perform inference on an image using a PyTorch model.
    See documentation for full parameter description.
    """
    assert len(depths_ims) == 3 or depths_ims.shape[-1] == 3, "depths_ims must contain three images."
    
    if get_bbox:
        assert rcnn_model is not None, "rcnn_model must be provided if get_bbox is True."
        assert totensor, "Image must be converted to a tensor if get_bbox is True."
        assert resize, "Image must be resized if get_bbox is True."
        depths_ims = ExtractEmbFrame(depths_ims[0], depths_ims[1], depths_ims[2], rcnn_model, device)
        depths_ims = [depths_ims[0], depths_ims[1], depths_ims[2]]
    
    if isinstance(depths_ims, List):
        image = np.stack(depths_ims, axis=-1)
    else:
        image = depths_ims
    
    if totensor:
        image = transforms.ToTensor()(image)
    if resize:
        image = transforms.Resize((224, 224))(image)
    if normalize:
        image /= 255.0
    
    # Add a batch dimension
    image = image.unsqueeze(0).to(device)
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image).squeeze(0)
        
    
    if map_output or output_to_str:
        output = torch.argmax(output).item()
        if output_to_str:
            output = mapping[output]
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on image(s) with specified focal depths.")
    # Mutually exclusive options: single-image, timelapse, or three separate focal images
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single-image",
        type=str,
        help="Path to a single image. The image will be duplicated as an RGB image across all channels."
    )
    group.add_argument(
        "--timelapse-dir",
        type=str,
        help="Path to a directory of images (each image is a timepoint) OR a directory containing exactly 3 subdirectories, "
             "each subdirectory containing images corresponding to a focal depth."
    )
    group.add_argument(
        "--F_neg15",
        type=str,
        help="Path to the F-15 focal depth image."
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
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=available_models,
        help=f"Name of the model to load from available models: {available_models}"
    )

    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="If provided, postprocess the model output (map raw output to class labels).",
        default=True
    )
    
    args = parser.parse_args()
    
    device = get_device()
    
    # Load the models
    model_class = mapping[args.model_name][0]
    model_class_arg = mapping[args.model_name][1]  # although not used in load_model below
    model_path = os.path.join(MODELS_DIR, f"{args.model_name}.pth")
    rcnn_model, rcnn_device = load_faster_RCNN_model_device(RCNN_PATH)
    # For regular inference we still map output to class label.
    # For timelapse we want the raw output (probability vector), so we use map_output=False.
    # find class 
    model, epoch, best_val_auc = load_model(model_path, get_device(), NCLASS, model_class=model_class)
    
    
    # List to store inference outputs
    outputs = []
    
    if args.timelapse_dir:
        timelapse_dir = args.timelapse_dir
        if not os.path.isdir(timelapse_dir):
            print(f"Error: {timelapse_dir} is not a valid directory.")
            exit(1)
        # Check for subdirectories
        subdirs = sorted([d for d in os.listdir(timelapse_dir) if os.path.isdir(os.path.join(timelapse_dir, d))])
        if len(subdirs) == 3:
            # Assume each subdirectory corresponds to a focal depth.
            focal_paths = [os.path.join(timelapse_dir, d) for d in subdirs]
            # List timepoints by intersecting sorted filenames across subdirectories.
            timepoint_files = []
            list_files = [sorted(os.listdir(fp)) for fp in focal_paths]
            # Determine minimum count across subdirectories
            num_timepoints = min(len(files) for files in list_files)
            for i in tqdm(range(num_timepoints)):
                file_paths = [os.path.join(focal_paths[j], list_files[j][i]) for j in range(3)]
                # Load images
                images = []
                for fp in file_paths:
                    img = cv2.imread(fp)
                    if img is None:
                        print(f"Failed to load image at {fp}")
                        exit(1)
                    images.append(img)
                # For timelapse, we want the raw probability vector so we do not map output.
                outputs.append(inference(model, device, images, map_output=False, output_to_str=False, 
                                         rcnn_model=rcnn_model))
        elif len(subdirs) > 0 and len(subdirs) != 3:
            print(f"Error: {timelapse_dir} must contain exactly 3 subdirectories for focal depth images. This model uses three focal depths.")
            exit(1)
        else:
            # Assume directory contains images: each image is a timepoint and will be duplicated to form the three channels.
            timepoint_files = sorted([f for f in os.listdir(timelapse_dir) if os.path.isfile(os.path.join(timelapse_dir, f))])
            for file in tqdm(timepoint_files):
                fp = os.path.join(timelapse_dir, file)
                single_image = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if single_image is None:
                    print(f"Failed to load image at {fp}")
                    continue
                duplicated_image = cv2.cvtColor(single_image, cv2.COLOR_GRAY2RGB)
                outputs.append(inference(model, device, duplicated_image.transpose(2, 0, 1), map_output=False, output_to_str=False, 
                                         rcnn_model=rcnn_model))
        
        np.save("raw_timelapse_outputs.npy", np.array(outputs))
        print("Raw outputs saved to raw_timelapse_outputs.npy")
        if args.postprocess:
            outputs = monotonic_decoding(np.array(outputs), loss='NLL')
            print("Postprocessed outputs saved to postprocessed_timelapse_outputs.npy")
        max_prob_classes = [np.argmax(output) for output in outputs]
        np.savetxt("max_prob_classes.csv", max_prob_classes, delimiter=",")
        print("Max probability classes saved to max_prob_classes.csv")


        # plot the max_prob_classes over time (timepoints) and save
        plt.plot(max_prob_classes)
        plt.xlabel("Timepoints")
        plt.ylabel("Max Probability Class")
        plt.title("Max Probability Class Over Time")
        plt.show()
        plt.savefig("max_prob_classes.png")
        print("Plot saved to max_prob_classes.png")


    elif args.single_image:
        # Single image: duplicate grayscale image as RGB
        single_image = cv2.imread(args.single_image, cv2.IMREAD_GRAYSCALE)
        if single_image is None:
            print(f"Failed to load image at {args.single_image}")
            exit(1)
        duplicated_image = cv2.cvtColor(single_image, cv2.COLOR_GRAY2RGB)
        depths_ims = [duplicated_image, duplicated_image, duplicated_image]
        # For single image we still map output to class label
        output = inference(model, get_device(), depths_ims, rcnn_model=rcnn_model, output_to_str=True)
        print(f"Class label: {output}")
    else:
        # Three separate images: all three paths must be provided.
        if not all([args.F_neg15, args.F0, args.F15]):
            print("Error: When not using --single-image or --timelapse-dir, all three focal depth images (--F_neg15, --F0, --F15) must be provided.")
            exit(1)
        image_F_neg15 = cv2.imread(args.F_neg15)
        image_F0 = cv2.imread(args.F0)
        image_F15 = cv2.imread(args.F15)
        
        if image_F_neg15 is None:
            print(f"Failed to load image at {args.F_neg15}")
            exit(1)
        if image_F0 is None:
            print(f"Failed to load image at {args.F0}")
            exit(1)
        if image_F15 is None:
            print(f"Failed to load image at {args.F15}")
            exit(1)
            
        depths_ims = [image_F_neg15, image_F0, image_F15]
        output = inference(model, get_device(), depths_ims, rcnn_model=rcnn_model, output_to_str=True)
        print(f"Class label: {output}")