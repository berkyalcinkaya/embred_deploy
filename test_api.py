import sys
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64
from io import BytesIO
import time

def decode_base64_img(b64str):
    img_bytes = base64.b64decode(b64str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img

def main(image_path):
    api_url = "http://localhost:8000/predict/timepoint"
    model_name = "CustomResNet50Unfrozen_CE_balanced_embSplits"  # <-- Replace with a valid model name

    times = {}
    # Read image as bytes (not included in timing)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Serialization: Prepare request (not including file read)
    t_serial_start = time.time()
    files = {
        "single_image": ("image.png", image_bytes, "image/png"),
    }
    data = {
        "model_name": model_name,
        "return_bbox": "true"
    }
    t_serial_end = time.time()
    times['serialization'] = t_serial_end - t_serial_start

    # Inference: Send request to API
    t_inf_start = time.time()
    response = requests.post(api_url, files=files, data=data)
    t_inf_end = time.time()
    times['inference'] = t_inf_end - t_inf_start
    response.raise_for_status()

    # Deserialization: Decode response and image (not including plotting or loading input image)
    t_deserial_start = time.time()
    result = response.json()
    bbox_b64 = result["bbox_crops"]["r"]
    bbox_crop = decode_base64_img(bbox_b64)
    t_deserial_end = time.time()
    times['deserialization'] = t_deserial_end - t_deserial_start

    # Plot (not included in timing)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    axs[1].imshow(bbox_crop, cmap='gray')
    axs[1].set_title(result["predicted_label"])
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    print("\nTiming (seconds):")
    for k, v in times.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_api.py path/to/image.png")
        sys.exit(1)
    main(sys.argv[1])
