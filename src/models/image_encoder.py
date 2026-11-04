import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Optional

class ImageEncoder(nn.Module):
    """
    Image encoder using a pretrained Vision Transformer (ViT).
    """
    def __init__(
        self,
        checkpoint: str = "google/vit-base-patch16-224",
        freeze: bool = True
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(checkpoint)

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)

        # ViT typically uses the [CLS] token for global representation
        # pooler_output: [batch, hidden_dim]
        return outputs.pooler_output

    def forward_patches(self, pixel_values):
        """
        Returns the patch embeddings for cross-attention.
        Returns: [batch, num_patches + 1, hidden_dim]
        """
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state
