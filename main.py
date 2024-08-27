import os
import zipfile
from tqdm import tqdm

def zip_current_folder(zip_name="zipped_folder.zip"):
    """
    Zips all files in the current folder into a zip file with a progress bar.

    :param zip_name: The name of the output zip file.
    """
    # Get the current folder path
    current_folder = os.getcwd()

    # Collect all files in the current folder
    files_to_zip = []
    for root, dirs, files in os.walk(current_folder):
        for file in files:
            # Get the full file path
            file_path = os.path.join(root, file)
            files_to_zip.append(file_path)

    # Create a zip file
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Use tqdm for the progress bar
        for file_path in tqdm(files_to_zip, desc="Zipping files", unit="file"):
            # Add the file to the zip file, relative to the current folder
            zipf.write(file_path, os.path.relpath(file_path, current_folder))

    print(f"All files in the current folder have been zipped into {zip_name}")

if __name__ == "__main__":
    zip_current_folder("output.zip")
