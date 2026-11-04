# src/data/fakeddit_dataset.py

"""
Dataset adapter for the Fakeddit subset.
Provides a PyTorch ``torch.utils.data.Dataset`` that yields tokenized text,
pre‑processed image tensors, and binary labels.

Assumptions:
- ``csv_path`` points to a CSV with at least the columns ``title`` (text), ``image_url`` (URL or filename), and ``label`` (0/1).
- Images are stored locally under ``image_root`` (e.g., ``data/raw/images``).
- ``image_url`` may be a full URL; we extract the basename and look for that file in ``image_root``.
- Missing images are replaced with a zero‑tensor of the appropriate shape.
"""

import os
import pandas as pd
from typing import Any, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.features.emotion_lexicon import nrc_emotion_vector
from src.features.distilbert_emotion import distilbert_emotion_embedding

class FakedditDataset(Dataset):
    """PyTorch Dataset for the Fakeddit subset.

    Parameters
    ----------
    csv_path: str
        Path to the processed CSV file (e.g., ``fakeddit_processed.csv``).
    image_root: str
        Directory containing the downloaded images.
    tokenizer_name: str, optional
        Name of the HuggingFace tokenizer for the text encoder (default ``roberta-base``).
    max_seq_len: int, optional
        Maximum sequence length for tokenization (default 128).
    vit_checkpoint: str, optional
        ViT checkpoint name used for the image feature extractor (default ``google/vit-base-patch16-224``).
    transform: callable, optional
        Optional additional transform applied to the image tensor after ViT preprocessing.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        tokenizer_name: str = "roberta-base",
        max_seq_len: int = 128,
        vit_checkpoint: str = "google/vit-base-patch16-224",
        transform: Any = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        # ViTFeatureExtractor may not be available; handle gracefully
        try:
            from transformers import ViTFeatureExtractor
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(vit_checkpoint)
        except Exception:
            self.feature_extractor = None
        self.transform = transform

        # Verify required columns exist
        required_cols = {"title", "image_url", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, url: str) -> torch.Tensor:
        # Guard against missing or NaN URLs which pandas may represent as float
        if not isinstance(url, str) or pd.isna(url):
            return torch.zeros(3, 224, 224, dtype=torch.float32)
        # Extract filename from URL (handles both full URLs and simple filenames)
        filename = os.path.basename(url)
        image_path = os.path.join(self.image_root, filename)
        if not os.path.isfile(image_path):
            # Image missing – return a zero‑tensor placeholder (same shape as ViT expects)
            return torch.zeros(3, 224, 224, dtype=torch.float32)
        try:
            img = Image.open(image_path).convert("RGB")
            encoded = self.feature_extractor(images=img, return_tensors="pt")
            pixel_values = encoded["pixel_values"].squeeze(0)  # shape [3, 224, 224]
            if self.transform:
                pixel_values = self.transform(pixel_values)
            return pixel_values
        except Exception as e:
            print(f"[FakedditDataset] Warning: could not load image {image_path}: {e}")
            return torch.zeros(3, 224, 224, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row["title"]) if not pd.isna(row["title"]) else ""
        # Tokenize text for the main model (BERT/CLIP tokenizer)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values = self._load_image(row["image_url"])
        label = int(row["label"])
        # ---- Emotion features ----
        # NRC lexicon vector (10‑dim numpy -> torch FloatTensor)
        lex_vec = torch.from_numpy(nrc_emotion_vector(text)).float()
        # DistilBERT emotion embedding (768‑dim tensor)
        distil_vec = distilbert_emotion_embedding(text).float()
        return {
            "raw_text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
            "lexicon_emotion": lex_vec,
            "distil_emotion": distil_vec,
        }

def get_fakeddit_dataset(
    csv_path: str,
    image_root: str,
    tokenizer_name: str = "roberta-base",
    max_seq_len: int = 128,
    vit_checkpoint: str = "google/vit-base-patch16-224",
) -> FakedditDataset:
    return FakedditDataset(
        csv_path=csv_path,
        image_root=image_root,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        vit_checkpoint=vit_checkpoint,
    )
