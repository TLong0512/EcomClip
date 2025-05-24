import os
import zipfile
import argparse
import wget
from pathlib import Path

def download_and_extract_dataset(file_url: str):
    # Create directory for dataset
    dataset_dir = Path("image_dataset")
    dataset_dir.mkdir(exist_ok=True)

    # Download the ZIP file from the given URL using wget
    print(f"Downloading from {file_url}")
    output_path = wget.download(file_url, out="temp_dataset.zip")

    # Extract the ZIP file into the dataset directory
    print("Unzipping dataset...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    print(f"Dataset has been unzipped into: {dataset_dir.resolve()}")

    # Remove the ZIP file after extraction (optional)
    os.remove(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract ZIP from a direct URL")
    parser.add_argument("--url", required=True, help="Direct URL to the ZIP file (e.g. 'https://huggingface.co/datasets/username/dataset/resolve/main/file.zip')")
    
    args = parser.parse_args()

    download_and_extract_dataset(args.url)
