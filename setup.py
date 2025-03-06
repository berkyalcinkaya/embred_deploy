from setuptools import setup, find_packages

setup(
    name="embpred_deploy",
    version="0.1.0",
    description="Emb Deployment Package for Image Inference",
    author="Berk Yalcinkaya",
    author_email="berkyalc@stanford.edu",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "matplotlib",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "embpred_deploy=embpred_deploy.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
    ],
)