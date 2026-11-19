"""Optional CLIP claim-image consistency scoring, separate from classification."""

from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPConsistencyScorer:
    def __init__(
        self,
        checkpoint: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(checkpoint)
        self.model = CLIPModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, claim: str, image: Image.Image) -> float:
        inputs = self.processor(
            text=[claim],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        text_features = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        image_features = self.model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return float((text_features * image_features).sum().cpu())
