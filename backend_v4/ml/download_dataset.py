"""
SkyGuard AI - Dataset Downloader
Downloads and extracts the BGC-Jena Weather Station dataset (2017-2024).
"""
import os
import io
import zipfile
import requests

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "matthewjansen/bgc-jena-weather-station-dataset-20172024"
)

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ML_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "jena_climate.csv")


def download_jena_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 1024 * 1024:
        print(f"Dataset already exists at: {CSV_PATH} ({os.path.getsize(CSV_PATH) / 1024 / 1024:.2f} MB)")
        return CSV_PATH

    print("=" * 60)
    print("SkyGuard AI - Downloading Jena Weather Station Dataset")
    print("URL:", DATASET_URL)
    print("=" * 60)

    response = requests.get(DATASET_URL, timeout=300, stream=True)
    response.raise_for_status()

    content = response.content
    print(f"Downloaded ZIP ({len(content) / 1024 / 1024:.2f} MB). Extracting...")

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        files = z.namelist()
        print("Files in archive:", files)

        csv_files = [f for f in files if f.lower().endswith(".csv") and "jena_climate" in f.lower()]
        if not csv_files:
            csv_files = [f for f in files if f.lower().endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded archive.")

        target_file = csv_files[0]
        print("Extracting:", target_file)
        z.extract(target_file, DATA_DIR)

        extracted_path = os.path.join(DATA_DIR, target_file)
        if extracted_path != CSV_PATH:
            if os.path.exists(CSV_PATH):
                os.remove(CSV_PATH)
            os.rename(extracted_path, CSV_PATH)

    print(f"Extracted dataset saved at: {CSV_PATH}")
    print(f"Size: {os.path.getsize(CSV_PATH) / 1024 / 1024:.2f} MB")
    return CSV_PATH


if __name__ == "__main__":
    download_jena_dataset()
