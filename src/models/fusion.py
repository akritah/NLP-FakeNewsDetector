import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultimodalFusion(nn.Module):
    """
    Fusion module implemented for the Replication Paper.
    Focuses on strict feature concatenation of Semantic, Emotion, and Visual vectors.
    """
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        fusion_method: str = "concat",
        shared_dim: int = 256
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.shared_dim = shared_dim

        # Projections to a shared dimension for concatenation
        self.text_proj = nn.Linear(text_dim, shared_dim) if text_dim != shared_dim else nn.Identity()
        self.image_proj = nn.Linear(image_dim, shared_dim) if image_dim != shared_dim else nn.Identity()
        if fusion_method == "cross_attention":
            self.text_to_image = nn.MultiheadAttention(shared_dim, num_heads=8, batch_first=True)
            self.image_to_text = nn.MultiheadAttention(shared_dim, num_heads=8, batch_first=True)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        emotion_features: torch.Tensor = None
        , text_seq: torch.Tensor = None
        , image_seq: torch.Tensor = None
        , text_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Strict concatenation of modalities.
        Returns: [batch, text_shared + image_shared + emotion_dim]
        """
        t = self.text_proj(text_features)
        i = self.image_proj(image_features)

        if self.fusion_method == "cross_attention":
            if text_seq is None or image_seq is None:
                raise ValueError("cross_attention requires text_seq and image_seq.")
            text_tokens = self.text_proj(text_seq)
            image_tokens = self.image_proj(image_seq)
            key_padding_mask = None
            if text_mask is not None:
                key_padding_mask = text_mask.squeeze(-1).eq(0)
            text_attended, _ = self.text_to_image(
                text_tokens, image_tokens, image_tokens
            )
            image_attended, _ = self.image_to_text(
                image_tokens, text_tokens, text_tokens, key_padding_mask=key_padding_mask
            )
            t = text_attended.mean(dim=1)
            i = image_attended.mean(dim=1)

        if emotion_features is not None:
            # Paper's proposed approach: Concatenate all three
            return torch.cat([t, i, emotion_features], dim=-1)

        return torch.cat([t, i], dim=-1)
