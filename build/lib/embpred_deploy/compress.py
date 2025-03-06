import os
import zipfile
import shutil
from .config import MODELS_DIR

def compress_model_files(models_dir, output_zip_path):
    """
    Compress all .pt and .pth files in models_dir into a zip archive.
    """
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth'):
                    filepath = os.path.join(root, file)
                    # Store the file relative to the models_dir for cleaner archive structure
                    arcname = os.path.relpath(filepath, models_dir)
                    zipf.write(filepath, arcname)
    print(f"Created zip archive: {output_zip_path}")

if __name__ == "__main__":
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the output zip file path in the current directory
    output_zip_name = "model_files.zip"
    output_zip_path = os.path.join(script_dir, output_zip_name)
    
    # Compress model files
    compress_model_files(MODELS_DIR, output_zip_path)
    
    # Move the zip file to the macOS Downloads folder
    downloads_dir = os.path.expanduser("~/Downloads")
    destination_zip_path = os.path.join(downloads_dir, output_zip_name)
    
    try:
        shutil.move(output_zip_path, destination_zip_path)
        print(f"Moved zip archive to: {destination_zip_path}")
    except Exception as e:
        print(f"Error moving zip archive: {e}")