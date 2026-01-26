import cv2
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def extract_emb_frame_2d(embframe, model, device):# what is return type of this function? 
    return ExtractEmbFrame(embframe, embframe, embframe, model, device)[0]

def ExtractEmbFrame(r_channel, g_channel, b_channel, model, device):
    bbox = get_emb_frame_bbox(g_channel, model, device)
    return crop_with_bbox(r_channel, g_channel, b_channel, bbox)

def get_emb_frame_bbox(im_2D, model, device):
    
    r_rgb = cv2.cvtColor(im_2D, cv2.COLOR_GRAY2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensor = transform(r_rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        predictions = model(image_tensor)
        
        best_bbox = None
        best_score = 0
        for bbox, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
            if score > best_score:
                best_bbox = bbox
                best_score = score

    if best_bbox is None:
        return 
    else:
        best_bbox = best_bbox.cpu().numpy()
        x_min, y_min, x_max, y_max = best_bbox.astype(int)
        return x_min, y_min, x_max, y_max
    
def crop_with_bbox(r_channel, g_channel, b_channel, bbox, size=(800, 800)):
    if bbox is None:
        padded_r = np.zeros(size, dtype=np.uint8) # update the size
        padded_g = padded_r
        padded_b = padded_r
    else:
        x_min, y_min, x_max, y_max = bbox

        cropped_r = r_channel[y_min:y_max, x_min:x_max]
        cropped_g = g_channel[y_min:y_max, x_min:x_max]
        cropped_b = b_channel[y_min:y_max, x_min:x_max]

        h, w = cropped_r.shape

        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            pad_top = pad_bottom = 0
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            pad_left = pad_right = 0

        padded_r = cv2.copyMakeBorder(
            cropped_r,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )
        padded_g = cv2.copyMakeBorder(
            cropped_g,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )
        padded_b = cv2.copyMakeBorder(
            cropped_b,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )

        return padded_r, padded_g, padded_b

