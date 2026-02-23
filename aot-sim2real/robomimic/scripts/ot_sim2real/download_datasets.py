import sys
import os

DATASET_DIR = None  # Specify the dataset directory here, e.g., "/path/to/dataset"

def download_folder(url, output_dir):
    """
    Downloads a Google Drive folder using the gdown library.
    Existing files are skipped (resumable).
    """
    try:
        import gdown
    except ImportError:
        print("Error: Downloading a folder requires the 'gdown' library.")
        print("Please run: pip install gdown")
        input("Press Enter to exit...")
        sys.exit(1)

    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Starting download from: {url}")
    print(f"Saving to: {output_dir}")
    print("-" * 40)
    
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    
    print("-" * 40)
    print(f"Download complete! Files saved in: {output_dir}")

if __name__ == "__main__":
    FOLDER_URL = "https://drive.google.com/drive/folders/1ZzAVSgeie2886xeIc-EQ88tjTq0ICtIP?usp=sharing"
    
    print("--- Dataset Downloader ---")
    
    if DATASET_DIR is None:
        destination_dir = input("Enter the destination directory to save the dataset: ").strip()
    else:
        destination_dir = DATASET_DIR
    
    download_folder(FOLDER_URL, destination_dir)