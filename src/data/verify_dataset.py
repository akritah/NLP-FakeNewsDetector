import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_verified_dataset(csv_path, image_dir, output_path):
    """
    Filters the original CSV to keep only rows where the image exists on disk.
    """
    logger.info(f"Loading original CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Get list of all files actually present in the image directory
    if not os.path.exists(image_dir):
        logger.error(f"Image directory {image_dir} does not exist!")
        return

    existing_images = set(os.listdir(image_dir))
    logger.info(f"Found {len(existing_images)} images on disk.")

    if "id" not in df.columns:
        raise ValueError("The CSV must contain an id column matching downloaded image names.")

    # Filter for images that exist and have a size > 0
    def local_image_name(sample_id):
        if pd.isna(sample_id):
            return None
        return f"{sample_id}.jpg"

    def check_image_exists(sample_id):
        image_name = local_image_name(sample_id)
        if image_name is None:
            return False
        path = os.path.join(image_dir, image_name)
        return image_name in existing_images and os.path.getsize(path) > 0

    logger.info("Filtering CSV for existing images...")
    mask = df["id"].apply(check_image_exists)
    verified_df = df.loc[mask].copy()
    verified_df["image_url"] = verified_df["id"].map(local_image_name)

    logger.info(f"Verified {len(verified_df)} samples with valid images.")

    verified_df.to_csv(output_path, index=False)
    logger.info(f"Saved verified dataset to {output_path}")

if __name__ == "__main__":
    create_verified_dataset(
        csv_path="data/raw/fakeddit_processed.csv",
        image_dir="data/raw/images/",
        output_path="data/raw/fakeddit_verified.csv"
    )
