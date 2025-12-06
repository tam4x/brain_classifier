import os
import shutil
import zipfile
from pathlib import Path

# -------------------------
#  Dataset Info (Kaggle IDs)
# -------------------------
DATASETS = {
    "alzheimer": "borhanitrash/alzheimer-mri-disease-classification-dataset",
    "tumor_a": "gabrielleyva307/brain-tumor-mri-classification-dataset-2025",
    "tumor_b": "pkdarabi/medical-image-dataset-brain-tumor-detection"
}

TARGET_DIR = "data"

import os
import pandas as pd
import base64
from PIL import Image
from io import BytesIO

def process_alzheimer_parquet(parquet_train, parquet_test, output_dir="data/alzheimer"):
    """
    Combines train + test parquet files and writes each image into:
    
    data/alzheimer/yes/*.jpg
    data/alzheimer/no/*.jpg
    """

    # Create output folders
    yes_dir = os.path.join(output_dir, "yes")
    no_dir = os.path.join(output_dir, "no")

    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)

    print("Loading Alzheimer parquet files...")

    # Load parquet data
    train_df = pd.read_parquet(parquet_train)
    test_df = pd.read_parquet(parquet_test)

    # Combine them
    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Total Alzheimer samples: {len(df)}")

    # Mapping: 0,1,3 -> YES ; 2 -> NO
    yes_labels = {0, 1, 3}

    print("Converting images...")

    for idx, row in df.iterrows():
        # decode base64 image
        img_bytes = base64.b64decode(row["image"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        label = row["label"]

        # assign global folder
        if label in yes_labels:
            out_path = os.path.join(yes_dir, f"alz_{idx}.jpg")
        else:
            out_path = os.path.join(no_dir, f"alz_{idx}.jpg")

        img.save(out_path)

    print("✔ Alzheimer data processed successfully!")
    print(f"YES images: {len(os.listdir(yes_dir))}")
    print(f"NO images:  {len(os.listdir(no_dir))}")


def download_and_prepare_datasets(kaggle_json_path="kaggle.json"):
    """Downloads 3 datasets and creates:
    
        data/
        ├── alzheimer/
        └── tumor/
            ├── Glioma/
            ├── Meningioma/
            ├── Pituitary/
            └── No Tumor/
    """

    # ------------------------------------
    # 1. Create target folder
    # ------------------------------------
    os.makedirs(TARGET_DIR, exist_ok=True)

    # ------------------------------------
    # 2. Setup Kaggle API
    # ------------------------------------
    print("Setting up Kaggle API...")
    os.makedirs("/root/.kaggle", exist_ok=True)
    shutil.copy(kaggle_json_path, "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 600)

    os.system("pip install kaggle --quiet")

    # ------------------------------------
    # 3. Download all 3 datasets
    # ------------------------------------
    for name, kaggle_id in DATASETS.items():
        print(f"\nDownloading {name} dataset...")
        os.system(f"kaggle datasets download -d {kaggle_id} -p {TARGET_DIR}")

    # ------------------------------------
    # 4. Extract ZIP files
    # ------------------------------------
    print("\nExtracting datasets...")
    for file in os.listdir(TARGET_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(TARGET_DIR, file)
            print("Extract:", file)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(TARGET_DIR)
            os.remove(zip_path)

    ## Creating Alzheimer folder
    
    # ------------------------------------
    # 6. Create combined tumor folder
    # ------------------------------------
    print("Setting up Tumor dataset...")
    final_tumor = os.path.join(TARGET_DIR, "tumor")
    os.makedirs(final_tumor, exist_ok=True)

    tumor_classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    # make subfolders
    for c in tumor_classes:
        os.makedirs(os.path.join(final_tumor, c), exist_ok=True)

    # ------------------------------------
    # 7. Merge tumor dataset A + B
    # ------------------------------------
    for folder in os.listdir(TARGET_DIR):
        if "tumor" in folder.lower() or "brain" in folder.lower():
            src_path = os.path.join(TARGET_DIR, folder)

            # search for known class folders
            for root, dirs, _ in os.walk(src_path):
                for d in dirs:
                    if d in tumor_classes:
                        src_dir = os.path.join(root, d)
                        dst_dir = os.path.join(final_tumor, d)

                        for file in os.listdir(src_dir):
                            shutil.move(os.path.join(src_dir, file), dst_dir)

            shutil.rmtree(src_path, ignore_errors=True)

    print("\n✔ DONE! Final structure created:")
    print("data/alzheimer")
    print("data/tumor")


if __name__ == "__main__":
    download_and_prepare_datasets()
