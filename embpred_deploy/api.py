# type: ignore
from fastapi import FastAPI, File, UploadFile, Form, HTTPException  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from typing import Optional, List
from pathlib import Path
import numpy as np
import cv2
import torch
from embpred_deploy.rcnn import ExtractEmbFrame
from embpred_deploy.utils import load_model, get_device, class_mapping
from embpred_deploy.models.mapping import model_mapping
from embpred_deploy.main import NCLASS
from embpred_deploy.config import MODELS_DIR # this is for testing only
import io
import base64
import time

app = FastAPI()

# Dynamically determine file path inside Docker container
#MODEL_DIR = Path(__file__).parent
# cast MODELS_DIR to Path
RCNN_PATH = MODELS_DIR / "rcnn.pt"

# Keep models in memory for repeated calls
rcnn_model = None
rcnn_device = None
timepoint_model = None
timepoint_model_name = None
timepoint_model_class = None
timepoint_model_class_args = None
timepoint_model_device = None

def load_rcnn():
    global rcnn_model, rcnn_device
    if rcnn_model is None:
        print("[RCNN] Loading RCNN model from:", RCNN_PATH)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        rcnn_model_loaded = torch.load(RCNN_PATH, map_location=device, weights_only=False)
        rcnn_model = rcnn_model_loaded
        rcnn_device = device
        print(f"[RCNN] Model loaded on device: {device}")
    else:
        print("[RCNN] Using cached RCNN model.")
    return rcnn_model, rcnn_device

def load_timepoint_model(model_name: str):
    global timepoint_model, timepoint_model_name, timepoint_model_class, timepoint_model_class_args, timepoint_model_device
    if (timepoint_model is None) or (timepoint_model_name != model_name):
        print(f"[Timepoint] Loading timepoint model: {model_name}")
        device = get_device()
        model_class = model_mapping[model_name][0]
        model_class_arg = model_mapping[model_name][1]
        model_path = MODELS_DIR / f"{model_name}.pth"
        print(f"[Timepoint] Model path: {model_path}")
        model, epoch, best_val_auc = load_model(str(model_path), device, NCLASS, model_class=model_class, class_args=model_class_arg)
        timepoint_model = model
        timepoint_model_name = model_name
        timepoint_model_class = model_class
        timepoint_model_class_args = model_class_arg
        timepoint_model_device = device
        print(f"[Timepoint] Model loaded on device: {device}")
    else:
        print(f"[Timepoint] Using cached timepoint model: {model_name}")
    return timepoint_model, timepoint_model_device

def read_imagefile(file: UploadFile) -> np.ndarray:
    print(f"[Image] Reading image file: {file.filename}")
    image_bytes = file.file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Image] Could not decode image: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}")
    print(f"[Image] Image shape: {img.shape}")
    return img

def prepare_images(single_image: Optional[UploadFile], images: Optional[List[UploadFile]]) -> List[np.ndarray]:
    if single_image:
        print("[Prepare] Single image provided.")
        img = read_imagefile(single_image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print("[Prepare] Converted single image to RGB and split channels.")
        return [img_rgb[..., i] for i in range(3)]
    elif images and len(images) == 3:
        print("[Prepare] Three images provided.")
        return [read_imagefile(f) for f in images]
    else:
        print("[Prepare] Invalid image input. single_image:", single_image, "images:", images)
        raise HTTPException(status_code=400, detail="Provide either a single image or exactly three images.")

def encode_img_to_base64(img: np.ndarray) -> str:
    # Encode numpy image to PNG and then to base64 string
    _, buffer = cv2.imencode('.png', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64

@app.post("/predict/bbox")
async def predict_bbox(
    single_image: Optional[UploadFile] = File(None, description="Single grayscale image to be stacked as RGB."),
    images: Optional[List[UploadFile]] = File(None, description="Three grayscale images for focal depths (F-15, F0, F15). Order matters."),
):
    print("[API] /predict/bbox called.")
    print(f"[API] single_image: {single_image}, images: {images}")
    rcnn, device = load_rcnn()
    depths_ims = prepare_images(single_image, images)
    try:
        padded_r, padded_g, padded_b = ExtractEmbFrame(depths_ims[0], depths_ims[1], depths_ims[2], rcnn, device)
        print("[BBox] Extracted bounding box crops.")
    except Exception as e:
        print(f"[BBox] RCNN extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"RCNN extraction failed: {str(e)}")
    if single_image:
        bbox_crops = {"r": encode_img_to_base64(padded_r)}
        print("[BBox] Returning only 'r' channel crop.")
    else:
        bbox_crops = {
            "r": encode_img_to_base64(padded_r),
            "g": encode_img_to_base64(padded_g),
            "b": encode_img_to_base64(padded_b)
        }
        print("[BBox] Returning all three channel crops.")
    return JSONResponse({"bbox_crops": bbox_crops})

@app.post("/predict/timepoint")
async def predict_timepoint(
    model_name: str = Form(..., description="Name of the timepoint model to use."),
    single_image: Optional[UploadFile] = File(None, description="Single grayscale image to be stacked as RGB."),
    images: Optional[List[UploadFile]] = File(None, description="Three grayscale images for focal depths (F-15, F0, F15). Order matters."),
    return_bbox: bool = Form(True, description="Whether to return the bounding box crops as well.")
):
    print("[API] /predict/timepoint called.")
    print(f"[API] model_name: {model_name}, single_image: {single_image}, images: {images}, return_bbox: {return_bbox}")
    timings = {}
    t0 = time.time()
    if model_name not in model_mapping:
        print(f"[API] Invalid model_name: {model_name}")
        raise HTTPException(status_code=400, detail=f"Model name {model_name} not available.")
    t_rcnn_start = time.time()
    rcnn, rcnn_device = load_rcnn()
    t_rcnn_end = time.time()
    timings['rcnn_load'] = t_rcnn_end - t_rcnn_start
    t_model_start = time.time()
    model, device = load_timepoint_model(model_name)
    t_model_end = time.time()
    timings['timepoint_model_load'] = t_model_end - t_model_start
    t_prep_start = time.time()
    depths_ims = prepare_images(single_image, images)
    t_prep_end = time.time()
    timings['prepare_images'] = t_prep_end - t_prep_start
    t_bbox_start = time.time()
    try:
        padded_r, padded_g, padded_b = ExtractEmbFrame(depths_ims[0], depths_ims[1], depths_ims[2], rcnn, rcnn_device)
        print("[Timepoint] Extracted bounding box crops.")
    except Exception as e:
        print(f"[Timepoint] RCNN extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"RCNN extraction failed: {str(e)}")
    t_bbox_end = time.time()
    timings['bbox_extraction'] = t_bbox_end - t_bbox_start
    t_pred_start = time.time()
    image = np.stack([padded_r, padded_g, padded_b], axis=-1)
    from torchvision import transforms
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Resize((224, 224))(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor).squeeze(0)
        pred_class = torch.argmax(output).item()
        pred_label = class_mapping[int(pred_class)]
        print(f"[Timepoint] Model prediction: class={pred_class}, label={pred_label}")
    t_pred_end = time.time()
    timings['prediction'] = t_pred_end - t_pred_start
    t1 = time.time()
    timings['total_endpoint'] = t1 - t0
    print("[API] Timing (seconds):")
    for k, v in timings.items():
        print(f"  {k}: {v:.4f}")
    result = {
        "predicted_class": int(pred_class),
        "predicted_label": pred_label
    }
    if return_bbox:
        if single_image:
            result["bbox_crops"] = {"r": encode_img_to_base64(padded_r)}
            print("[Timepoint] Returning only 'r' channel crop.")
        else:
            result["bbox_crops"] = {
                "r": encode_img_to_base64(padded_r),
                "g": encode_img_to_base64(padded_g),
                "b": encode_img_to_base64(padded_b)
            }
            print("[Timepoint] Returning all three channel crops.")
    return JSONResponse(result) 

