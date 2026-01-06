#!/usr/bin/env python3
"""
Build and deploy script for embpred_deploy package to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Success: {result.stdout}")
    return True

def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")
    
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for dir_pattern in dirs_to_clean:
        for path in Path(".").glob(dir_pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed: {path}")
            elif path.is_file():
                path.unlink()
                print(f"Removed: {path}")

def build_package():
    """Build the package."""
    return run_command("python -m build", "Building package")

def check_package():
    """Check the built package."""
    return run_command("python -m twine check dist/*", "Checking package")

def upload_to_testpypi():
    """Upload to TestPyPI."""
    print("\nUploading to TestPyPI...")
    print("Running: python -m twine upload --repository testpypi dist/*")
    result = subprocess.run("python -m twine upload --repository testpypi dist/*", shell=True)
    return result.returncode == 0

def upload_to_pypi():
    """Upload to PyPI."""
    print("\nUploading to PyPI...")
    print("Running: python -m twine upload dist/*")
    result = subprocess.run("python -m twine upload dist/*", shell=True)
    return result.returncode == 0

def main():
    """Main deployment function."""
    print("embpred_deploy PyPI Deployment Script")
    print("=" * 40)
    
    # Check if required tools are installed
    try:
        import build
        import twine
    except ImportError:
        print("Error: Required packages not found.")
        print("Please install: pip install build twine")
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Build package
    if not build_package():
        print("Build failed!")
        sys.exit(1)
    
    # Check package
    if not check_package():
        print("Package check failed!")
        sys.exit(1)
    
    # Ask user which PyPI to upload to
    print("\nChoose upload destination:")
    print("1. TestPyPI (recommended for testing)")
    print("2. PyPI (production)")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if not upload_to_testpypi():
            print("Upload to TestPyPI failed!")
            sys.exit(1)
        print("\nPackage uploaded to TestPyPI successfully!")
        print("You can test installation with: pip install --index-url https://test.pypi.org/simple/ embpred_deploy")
        
    elif choice == "2":
        if not upload_to_pypi():
            print("Upload to PyPI failed!")
            sys.exit(1)
        print("\nPackage uploaded to PyPI successfully!")
        print("You can install with: pip install embpred_deploy")
        
    elif choice == "3":
        if not upload_to_testpypi():
            print("Upload to TestPyPI failed!")
            sys.exit(1)
        if not upload_to_pypi():
            print("Upload to PyPI failed!")
            sys.exit(1)
        print("\nPackage uploaded to both TestPyPI and PyPI successfully!")
        
    elif choice == "4":
        print("Deployment cancelled.")
        sys.exit(0)
        
    else:
        print("Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main() 