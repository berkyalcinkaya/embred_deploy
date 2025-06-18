# type: ignore
from fastapi import FastAPI, File, UploadFile, Form, HTTPException  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from typing import Optional, List
from pathlib import Path
import numpy as np
import cv2
import torch
from embpred_deploy.rcnn import ExtractEmbFrame
from embpred_deploy.utils import mapping as model_mapping, load_model, get_device
from embpred_deploy.main import class_mapping, NCLASS
import io

app = FastAPI()

# Dynamically determine file path inside Docker container
MODEL_DIR = Path(__file__).parent
RCNN_PATH = MODEL_DIR / "rcnn.pt"

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
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        rcnn_model_loaded = torch.load(RCNN_PATH, map_location=device, weights_only=False)
        rcnn_model = rcnn_model_loaded
        rcnn_device = device
    return rcnn_model, rcnn_device

def load_timepoint_model(model_name: str):
    global timepoint_model, timepoint_model_name, timepoint_model_class, timepoint_model_class_args, timepoint_model_device
    if (timepoint_model is None) or (timepoint_model_name != model_name):
        device = get_device()
        # model_mapping keys are strings (model names)
        model_class = model_mapping[model_name][0]
        model_class_arg = model_mapping[model_name][1]
        model_path = MODEL_DIR / f"{model_name}.pth"
        model, epoch, best_val_auc = load_model(str(model_path), device, NCLASS, model_class=model_class, class_args=model_class_arg)
        timepoint_model = model
        timepoint_model_name = model_name
        timepoint_model_class = model_class
        timepoint_model_class_args = model_class_arg
        timepoint_model_device = device
    return timepoint_model, timepoint_model_device

def read_imagefile(file: UploadFile) -> np.ndarray:
    image_bytes = file.file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}")
    return img

def prepare_images(single_image: Optional[UploadFile], images: Optional[List[UploadFile]]) -> List[np.ndarray]:
    if single_image:
        img = read_imagefile(single_image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Return as three channels
        return [img_rgb[..., i] for i in range(3)]
    elif images and len(images) == 3:
        return [read_imagefile(f) for f in images]
    else:
        raise HTTPException(status_code=400, detail="Provide either a single image or exactly three images.")

@app.post("/predict/bbox")
async def predict_bbox(
    single_image: Optional[UploadFile] = File(None, description="Single grayscale image to be stacked as RGB."),
    images: Optional[List[UploadFile]] = File(None, description="Three grayscale images for focal depths (F-15, F0, F15). Order matters."),
):
    rcnn, device = load_rcnn()
    depths_ims = prepare_images(single_image, images)
    # Use ExtractEmbFrame to get bounding box crops
    try:
        padded_r, padded_g, padded_b = ExtractEmbFrame(depths_ims[0], depths_ims[1], depths_ims[2], rcnn, device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RCNN extraction failed: {str(e)}")
    # For API, return bbox crops as shapes
    return JSONResponse({
        "bbox_crops": {
            "r": list(np.array(padded_r).shape),
            "g": list(np.array(padded_g).shape),
            "b": list(np.array(padded_b).shape)
        }
    })

@app.post("/predict/timepoint")
async def predict_timepoint(
    model_name: str = Form(..., description="Name of the timepoint model to use."),
    single_image: Optional[UploadFile] = File(None, description="Single grayscale image to be stacked as RGB."),
    images: Optional[List[UploadFile]] = File(None, description="Three grayscale images for focal depths (F-15, F0, F15). Order matters."),
    return_bbox: bool = Form(True, description="Whether to return the bounding box crops as well.")
):
    if model_name not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Model name {model_name} not available.")
    rcnn, rcnn_device = load_rcnn()
    model, device = load_timepoint_model(model_name)
    depths_ims = prepare_images(single_image, images)
    # Get bbox crops
    try:
        padded_r, padded_g, padded_b = ExtractEmbFrame(depths_ims[0], depths_ims[1], depths_ims[2], rcnn, rcnn_device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RCNN extraction failed: {str(e)}")
    # Stack to 3-channel image
    image = np.stack([padded_r, padded_g, padded_b], axis=-1)
    # Prepare for model (ToTensor, Resize)
    from torchvision import transforms
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Resize((224, 224))(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor).squeeze(0)
        pred_class = torch.argmax(output).item()
        pred_label = class_mapping[int(pred_class)]
    result = {
        "predicted_class": int(pred_class),
        "predicted_label": pred_label
    }
    if return_bbox:
        result["bbox_crops"] = {
            "r": list(np.array(padded_r).shape),
            "g": list(np.array(padded_g).shape),
            "b": list(np.array(padded_b).shape)
        }
    return JSONResponse(result) 