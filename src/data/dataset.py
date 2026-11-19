import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakedditDataset(Dataset):
    """
    Dataset for Multimodal Fake News Detection using Fakeddit.
    Handles text and image pairs with associated labels.
    """
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        text_col: str,
        image_col: str,
        label_col: str,
        label_mapping: dict,
        tokenizer = None,
        image_processor = None,
        max_length: int = 128,
        subset_size: Optional[int] = None
    ):
        self.df = pd.read_csv(csv_path)

        if subset_size:
            logger.info(f"Using smoke test subset of size {subset_size}")
            self.df = self.df.sample(n=min(subset_size, len(self.df)), random_state=42).reset_index(drop=True)

        self.image_dir = image_dir
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.label_mapping = label_mapping
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        self._validate_and_clean_data()

    def _validate_and_clean_data(self):
        """Handle missing text, images, and invalid labels explicitly."""
        initial_len = len(self.df)

        # 1. Handle Missing Text
        self.df = self.df.dropna(subset=[self.text_col])

        # 2. Handle Invalid Labels
        # Keep only labels that are in our mapping
        self.df = self.df[self.df[self.label_col].astype(str).isin(self.label_mapping.keys())]

        # 3. Handle Missing/Corrupt Images
        # We check existence during __getitem__ to avoid scanning thousands of files at startup,
        # but we can do a basic check here if the image column is NaN.
        self.df = self.df.dropna(subset=[self.image_col])

        final_len = len(self.df)
        logger.info(f"Data cleaning: removed {initial_len - final_len} samples due to missing/invalid data.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[dict, int]:
        row = self.df.iloc[idx]

        # Text Processing
        text = str(row[self.text_col])
        if self.tokenizer:
            text_inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            # Squeeze to remove batch dimension [1, seq_len] -> [seq_len]
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        else:
            text_inputs = text

        # Image Processing
        img_path = os.path.join(self.image_dir, str(row[self.image_col]))
        try:
            image = Image.open(img_path).convert("RGB")
            if self.image_processor:
                image_inputs = self.image_processor(images=image, return_tensors="pt")
                image_inputs = {k: v.squeeze(0) for k, v in image_inputs.items()}
            else:
                image_inputs = image
        except Exception as e:
            raise RuntimeError(f"Unable to load required image {img_path}") from e

        # Label Mapping
        label_raw = str(row[self.label_col])
        label = self.label_mapping[label_raw]

        return {
            "text": text_inputs,
            "image": image_inputs,
            "raw_text": text,
            "id": row.name # Preserve stable identifier
        }, label
