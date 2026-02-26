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

**⚠️ REQUIRED**: Pretrained model weights are **not** included in the PyPI package. You must download them separately to use the package.

Model weights are stored in a **private AWS S3 bucket** named `cfai-model-weights`. You will need AWS credentials with permission to read from this bucket (ask the project maintainer for access).

### 1. Install the AWS CLI (securely)

Follow the official AWS instructions for your platform (see the [AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)). Example commands:

### 2. Authenticate with AWS

Configure your AWS credentials (access key, secret key, region) using:

```bash
aws configure
```

You can also use environment variables or an existing AWS profile; the important part is that the configured identity has `s3:GetObject` access to the `cfai-model-weights` bucket.

### 3. Download the model weights

First, find the `models` directory used by `embpred_deploy`:

```bash
python -c "from embpred_deploy.config import MODELS_DIR; print(MODELS_DIR)"
```

Then download the desired model into that directory. For the model
`New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64.pth`:

```bash
aws s3 cp \
  "s3://cfai-model-weights/New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64.pth" \
  /path/to/embpred_deploy/models/
```

Replace `/path/to/embpred_deploy/models/` with the path printed by the `MODELS_DIR` command above.

## Usage Instructions

To see all available CLI arguments and options, run:

```bash
embpred_deploy --help
```

If you are working from the source repository, you can also run:

```bash
python -m embpred_deploy.main --help
```

The inference script supports three main modes:


### 1. **Timelapse Inference**

Use the `--timelapse-dir` argument to process a sequence of images. This mode supports:

- **Single image per timepoint**: all images in one directory (each loaded in grayscale and duplicated to RGB).
- **Multiple focal depths per timepoint**: three subdirectories (each a focal depth); files are aligned by sorted filenames.

#### Example Command:

```bash
embpred_deploy \
  --timelapse-dir /path/to/timelapse_data \
  --model-name New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64 \
  --postprocess
```

Key outputs:
- Raw outputs: `raw_timelapse_outputs.npy`
- If `--postprocess` is enabled:
  - `max_prob_classes.csv`
  - `max_prob_classes.png`


### 2. **Single Image Inference**

Use the `--single-image` argument to run inference on a single image. The image is processed by duplicating its grayscale channel into RGB.

#### Example Command:

```bash
embpred_deploy \
  --single-image /path/to/image.jpg \
  --model-name New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64
```


### 3. **Three-Focal Depth Inference**

Provide three separate focal depth images using the `--F_neg15`, `--F0`, and `--F15` arguments.

#### Example Command:

```bash
embpred_deploy \
  --F_neg15 /path/to/F_neg15.jpg \
  --F0 /path/to/F0.jpg \
  --F15 /path/to/F15.jpg \
  --model-name New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64
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
