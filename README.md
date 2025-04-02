# embpred_deploy Documentation

## Overview

`embpred_deploy` is a deployment package for running inference using **Faster-RCNN and custom classification networks**. It supports various input modes, including **timelapse inference**, **single-image inference**, and **multi-focal depth inference**.


## Installation

### 1. Create and activate a Conda environment with Python 3.12

To ensure compatibility, create a new Conda environment:

```bash
conda create -n embd python=3.12
conda activate embd
```
### 2. Install `embpred_deploy` via pip (not supported yet)

Once the environment is set up, install the package:

```bash
pip install embpred_deploy
```

### 3. Install `embpred_deploy` via Git

Alternatively, if you prefer to pull the latest code directly from GitHub, run:

```bash
git clone https://github.com/berkyalcinkaya/embpred_deploy.git
cd embpred_deploy
pip install -e .
```


## Model Weights Installation

Pretrained model weights are stored on Google Drive. To use the latest trained models, download the weight files from:

[Google Drive Weights](https://drive.google.com/file/d/1bf7vCVUbkREmODBsdFPL7yA6ypg95tRT/view?usp=sharing)

### Install Weights

After downloading the zip file, run the installation script to extract and move the weight files into the appropriate `models` folder:

```bash
python -m embpred_deploy.install_weights /path/to/your/downloaded_weights.zip
```

This script:
- Unzips the archive
- Moves any `.pth` or `.pt` files to the `models` directory

## Usage Instructions

The inference script supports three modes:


### 1. **Timelapse Inference**

Use the `--timelapse-dir` argument to process a sequence of images. This mode supports two directory structures:

- **Single-image per timepoint**  
  - All images are stored in one directory.
  - Each image is loaded in grayscale and converted to RGB (duplicated channels).
  
- **Multiple focal depths per timepoint**  
  - The images must be organized into **three subdirectories**, each representing a different focal depth.
  - The script aligns images based on sorted filenames across subdirectories.

#### Example Command:

```bash
python -m embpred_deploy.main --timelapse-dir /path/to/your/timelapse_data --model-name YOUR_MODEL_NAME
```

#### Output:
- Raw outputs: `raw_timelapse_outputs.npy`
- If `--postprocess` is enabled:
  - `max_prob_classes.csv`
  - `max_prob_classes.png`


### 2. **Single Image Inference**

Use the `--single-image` argument to run inference on a single image. The image is processed by duplicating its grayscale channel into RGB.

#### Example Command:

```bash
python -m embpred_deploy.main --single-image /path/to/your/image.jpg --model-name YOUR_MODEL_NAME
```


### 3. **Three-Focal Depth Inference**

Provide three separate focal depth images using the `--F_neg15`, `--F0`, and `--F15` arguments.

#### Example Command:

```bash
python -m embpred_deploy.main --F_neg15 /path/to/F_neg15.jpg --F0 /path/to/F0.jpg --F15 /path/to/F15.jpg --model-name YOUR_MODEL_NAME
```


## Assumptions & Notes

### **Input Image Format**
- **Single image inference**:  
  - Image is loaded in grayscale and converted to 3-channel RGB.
- **Timelapse mode**:  
  - Image filenames must be **sorted** to ensure correct timepoint alignment.

### **Model Output**
- **Regular inference**:  
  - The script maps raw model output to class labels.
- **Timelapse inference**:  
  - Raw probability vectors are returned unless `--postprocess` is enabled.

### **Dependencies**
The package requires the following libraries (installed via pip or Conda):
- `pytorch`
- `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `tqdm`

Ensure these dependencies are installed in your environment before running inference.


## Support

For further details or troubleshooting, please refer to the source code or contact the maintainers.
