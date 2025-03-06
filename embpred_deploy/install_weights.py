# script: user specifies path to a zip file. Unzip it, move all files to models/import argparse
import zipfile
import os
import shutil
import argparse
import sys
from embpred_deploy.config import MODELS_DIR, PROJ_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Unzip a file and move all .pth files to the 'models' directory."
    )
    parser.add_argument(
        "zip_path",
        type=str,
        help="Path to the zip file containing model weights (.pth)."
    )
    args = parser.parse_args()

    zip_path = args.zip_path


    # Check that the zip file exists
    if not os.path.isfile(zip_path):
        print(f"Error: Zip file '{zip_path}' does not exist.")
        sys.exit(1)

    # Create models directory if it doesn't exist
    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Extract the zip file into a temporary directory
    temp_dir = os.path.join(PROJ_DIR, "temp_unzip")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zip file.")
        sys.exit(1)

    # Move .pth files from temp_dir to models_dir
    moved_any = False
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".pt"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(MODELS_DIR, file)
                shutil.move(src_file, dest_file)
                moved_any = True

    if not moved_any:
        print("No .pth files found in the provided zip archive.")

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    print("Done. Model weight files (if any) have been moved to the 'models' directory.")


if __name__ == "__main__":
    main()