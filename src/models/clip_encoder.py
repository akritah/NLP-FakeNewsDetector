import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPImageEncoder(nn.Module):
    """Frozen or trainable CLIP visual encoder for research experiments."""

    def __init__(
        self,
        checkpoint: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(checkpoint)

        if freeze:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip.get_image_features(pixel_values=pixel_values)

    def forward_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        patch_features = vision_outputs.last_hidden_state
        return self.clip.visual_projection(patch_features)
