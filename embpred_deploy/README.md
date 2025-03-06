
# Image Inference Script

## Overview

This script (`inf.py`) allows users to perform image classification using a pre-trained model. It supports two modes of operation:

1. **Single Image Mode**: Provide a single grayscale image, which is duplicated across three focal depths.
2. **Multiple Images Mode**: Provide three separate images corresponding to different focal depths: `F-15`, `F0`, and `F15`.

The script utilizes a Faster R-CNN model for object detection and a custom neural network (`BiggerNet3D224`) for classification.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Single Image Mode](#single-image-mode)
  - [Multiple Images Mode](#multiple-images-mode)
- [Inference Function](#inference-function)
  - [Function Signature](#function-signature)
  - [Parameters](#parameters)
  - [Returns](#returns)
  - [Example Usage](#example-usage)
- [Model Files](#model-files)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

- **Operating System**: macOS
- **Python Version**: 3.7 or higher
- **Hardware**: GPU (optional, for faster inference)

## Usage

### Single Image Mode

Allows the user to provide a single grayscale image. The script duplicates this image across three channels to create an RGB image for inference.

```bash
python inf.py --single-image path/to/image.jpg
```

**Example:**

```bash
python inf.py --single-image ./images/sample_grayscale.jpg
```

### Multiple Images Mode

Allows the user to provide three separate images corresponding to different focal depths: `F-15`, `F0`, and `F15`.

```bash
python inf.py --F_neg15 path/to/F_neg15.jpg --F0 path/to/F0.jpg --F15 path/to/F15.jpg
```

**Example:**

```bash
python inf.py --F_neg15 ./images/F_neg15.jpg --F0 ./images/F0.jpg --F15 ./images/F15.jpg
```

**Note:** You must provide either the `--single-image` argument or all three of `--F_neg15`, `--F0`, and `--F15`. These options are mutually exclusive.

## Inference Function

The `inference` function is responsible for processing the input images and returning the classification result.

### Function Signature

```python
def inference(
    model,
    device,
    depths_ims: Union[List[np.ndarray], torch.Tensor, np.ndarray],
    map_output=True,
    output_to_str=False,
    totensor=True,
    resize=True,
    normalize=True,
    get_bbox=True,
    rcnn_model=None,
    size=(224, 224)
):
    ...
```

### Parameters

- **model** (`torch.nn.Module`): The pre-trained classification model.
- **device** (`torch.device`): The device (`cpu` or `cuda`) on which computations will be performed.
- **depths_ims** (`List[np.ndarray] | torch.Tensor | np.ndarray`): A list of three images corresponding to different focal depths.
- **map_output** (`bool`, optional): If `True`, maps the output class index to its label using the `mapping` dictionary. Default is `True`.
- **output_to_str** (`bool`, optional): If `True`, returns the class label as a string. Default is `False`.
- **totensor** (`bool`, optional): If `True`, converts the image to a PyTorch tensor. Default is `True`.
- **resize** (`bool`, optional): If `True`, resizes the image to the specified `size`. Default is `True`.
- **normalize** (`bool`, optional): If `True`, normalizes the image by dividing by 255.0. Default is `True`.
- **get_bbox** (`bool`, optional): If `True`, uses the RCNN model to obtain bounding boxes before classification. Default is `True`.
- **rcnn_model** (`torch.nn.Module`, optional): The pre-trained Faster R-CNN model for object detection. Required if `get_bbox` is `True`.
- **size** (`tuple`, optional): The target size for resizing the image. Default is `(224, 224)`.

### Returns

- **output** (`int | str`): The predicted class index or class label, depending on the `map_output` and `output_to_str` flags.

### Example Usage

```python
import torch
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Union

# Assuming models are loaded and device is set
model, device = load_model(MODEL_PATH, get_device(), NCLASS)
rcnn_model, rcnn_device = load_faster_RCNN_model_device(RCNN_PATH)

# Load images
image_F_neg15 = cv2.imread('path/to/F_neg15.jpg')
image_F0 = cv2.imread('path/to/F0.jpg')
image_F15 = cv2.imread('path/to/F15.jpg')

depths_ims = [image_F_neg15, image_F0, image_F15]

# Run inference
output = inference(
    model=model,
    device=device,
    depths_ims=depths_ims,
    rcnn_model=rcnn_model,
    output_to_str=True
)

print(f"Predicted Class: {output}")
```

## Model Files

Ensure that the following model files are present in the specified paths:

- **Faster R-CNN Model**: `rcnn.pth`
- **Classification Model**: `model.pth`

**Note:** The paths are currently set as relative paths in the script. You may need to provide absolute paths or adjust them based on your directory structure.

## Troubleshooting

- **Model Loading Issues**:
  - Ensure that the model files (`rcnn.pth` and `model.pth`) exist in the specified paths.
  - Verify that the models are compatible with the PyTorch version installed.

- **Image Loading Errors**:
  - Check that the provided image paths are correct.
  - Ensure that the images are in a supported format (e.g., JPEG, PNG).

- **CUDA Errors**:
  - If running on a machine without a GPU, ensure that the `--use_GPU` flag is set to `False` or modify the script accordingly.

- **Missing Dependencies**:
  - Install any missing Python packages using `pip install`.

## License

This project is licensed under the MIT License.
