import os
import requests
import zipfile
import io

def download_file(url, target_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {target_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def main():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # 1. UCI Predict Students' Dropout and Academic Success
    # Direct URL for the zip file containing the data
    uci_url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
    uci_zip_path = os.path.join(raw_dir, "uci_dropout.zip")
    download_file(uci_url, uci_zip_path)
    
    if os.path.exists(uci_zip_path):
        with zipfile.ZipFile(uci_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "uci_dropout"))
        print("Extracted UCI Dropout dataset.")

    # 2. Mendeley AI Student Performance (using direct download link if possible)
    # Note: Mendeley links can be tricky, if this fails, user might need to download manually.
    mendeley_url = "https://data.mendeley.com/public-api/data/vzfyk22fhn/files/7b63f538-23ef-4f10-9f5b-1175c2763f35/content"
    mendeley_path = os.path.join(raw_dir, "mendeley_ai_course.csv")
    download_file(mendeley_url, mendeley_path)

    print("\n--- Manual Step Required ---")
    print("Please download the Kaggle 2024 dataset manually from:")
    print("https://www.kaggle.com/datasets/kamalnajem/student-performance-and-learning-behavior-dataset")
    print("And place the CSV file in: data/raw/kaggle_2024.csv")

if __name__ == "__main__":
    main()
