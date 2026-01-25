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

### 2. Install `embpred_deploy` via pip

#### Standard Installation (with GPU support)

The default installation includes PyTorch with CUDA support:

```bash
pip install embpred_deploy
```

#### CPU-Only Installation (lighter weight)

For a lighter-weight CPU-only installation (recommended if you don't need GPU support):

```bash
# Install the package without PyTorch dependencies
pip install embpred_deploy --no-deps

# Install CPU-only PyTorch and torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install opencv-python-headless>=4.5.0 numpy>=1.21.0 matplotlib>=3.3.0 tqdm>=4.60.0
```

Or as a one-liner:

```bash
pip install embpred_deploy --no-deps && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install opencv-python-headless numpy matplotlib tqdm
```

**Note**: The CPU-only installation is significantly smaller (~200MB vs ~2GB+) and is sufficient if you're running inference on CPU only.

**Important**: The PyPI package does not include model weights due to size limitations. You must download the model weights separately (see Model Weights Installation below).

### 3. Install `embpred_deploy` via Git

Alternatively, if you prefer to pull the latest code directly from GitHub, run:

```bash
git clone https://github.com/berkyalcinkaya/embpred_deploy.git
cd embpred_deploy
pip install -e .
```


## Model Weights Installation

**⚠️ REQUIRED**: Pretrained model weights are NOT included in the PyPI package due to size limitations. You must download them separately to use the package.

Model weights are stored on Google Drive. To use the latest trained models, download the weight files from:

[Google Drive Weights](https://drive.google.com/file/d/1c-BJ0dvxzaZ4wMxMlhiRtwzURz7_ryaH/view?usp=sharing)

### Install Weights

After downloading the zip file, run the installation script to extract and move the weight files into the appropriate `models` folder:

```bash
python -m embpred_deploy.install_weights /path/to/your/downloaded_weights.zip
```

This script:
- Unzips the archive
- Moves any `.pth` or `.pt` files to the `embpred_deploy/models` directory

**Note**: The `embpred_deploy/models` directory will be created automatically if it doesn't exist.

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
embpred_deploy --timelapse-dir /path/to/your/timelapse_data --model-name YOUR_MODEL_NAME
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
embpred_deploy --single-image /path/to/your/image.jpg --model-name YOUR_MODEL_NAME
```


### 3. **Three-Focal Depth Inference**

Provide three separate focal depth images using the `--F_neg15`, `--F0`, and `--F15` arguments.

#### Example Command:

```bash
embpred_deploy --F_neg15 /path/to/F_neg15.jpg --F0 /path/to/F0.jpg --F15 /path/to/F15.jpg --model-name YOUR_MODEL_NAME
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
- `pytorch` (or CPU-only version for lighter install)
- `torchvision` (or CPU-only version for lighter install)
- `opencv-python-headless`
- `numpy`
- `matplotlib`
- `tqdm`

**Note**: For CPU-only installations, use the CPU-only versions of PyTorch and torchvision as described in the Installation section above. This significantly reduces the package size.

Ensure these dependencies are installed in your environment before running inference.

## Development and Deployment

### Setting up for Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/berkyalcinkaya/embpred_deploy.git
cd embpred_deploy

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

### Deploying to PyPI

To deploy the package to PyPI:

1. **Install deployment tools:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run the deployment script:**
   ```bash
   python build_and_deploy.py
   ```

3. **Or manually build and deploy:**
   ```bash
   # Clean previous builds
   rm -rf build/ dist/ *.egg-info/
   
   # Build the package
   python -m build
   
   # Check the package
   python -m twine check dist/*
   
   # Upload to TestPyPI (recommended first)
   python -m twine upload --repository testpypi dist/*
   
   # Upload to PyPI (production)
   python -m twine upload dist/*
   ```

### PyPI Credentials

You'll need to set up your PyPI credentials. Create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = your_username
password = your_password

[testpypi]
repository = https://test.pypi.org/legacy/
username = your_username
password = your_password
```

## Support

For further details or troubleshooting, please refer to the source code or contact the maintainers.
