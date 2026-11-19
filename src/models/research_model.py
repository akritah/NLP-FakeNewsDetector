import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import MLPClassifier
from .fusion import MultimodalFusion
from .image_encoder import ImageEncoder
from .clip_encoder import CLIPImageEncoder
from .text_encoder import TextEncoder


class ResearchMultimodalModel(nn.Module):
    """Separate controlled experiment model; existing baseline models remain unchanged."""

    def __init__(self, config):
        super().__init__()
        model_config = config["model"]
        text_config = model_config["text_encoder"]

        self.text_encoder = TextEncoder(
            checkpoint=text_config["checkpoint"],
            freeze=text_config.get("freeze", True),
            pooling=text_config.get("pooling", "cls"),
        )
        text_dim = self.text_encoder.roberta.config.hidden_size

        visual_config = model_config.get("visual_encoder", {})
        visual_type = visual_config.get("type", "vit")
        self.visual_type = visual_type
        if visual_type == "clip":
            self.visual_encoder = CLIPImageEncoder(
                checkpoint=visual_config.get(
                    "checkpoint", "openai/clip-vit-base-patch32"
                ),
                freeze=visual_config.get("freeze", True),
            )
            visual_dim = self.visual_encoder.clip.config.projection_dim
        else:
            self.visual_encoder = ImageEncoder(
                checkpoint=visual_config.get(
                    "checkpoint", "google/vit-base-patch16-224"
                ),
                freeze=visual_config.get("freeze", True),
            )
            visual_dim = self.visual_encoder.vit.config.hidden_size

        self.fusion_method = model_config.get("fusion_method", "concat")
        shared_dim = model_config.get("hidden_dim", 256)
        self.configured_consistency = model_config.get(
            "cross_modal_consistency", False
        )
        consistency_dim = 1 if self.configured_consistency else 0
        if self.configured_consistency:
            self.text_consistency_projection = nn.Linear(text_dim, shared_dim)
            self.visual_consistency_projection = nn.Linear(visual_dim, shared_dim)

        if self.fusion_method in {"concat", "cross_attention"}:
            self.fusion = MultimodalFusion(
                text_dim=text_dim,
                image_dim=visual_dim,
                fusion_method=self.fusion_method,
                shared_dim=shared_dim,
            )
            if self.fusion_method == "text_only":
                input_dim = text_dim
            else:
                input_dim = shared_dim * 2
        elif self.fusion_method == "text_only":
            self.fusion = None
            input_dim = text_dim
        elif self.fusion_method == "image_only":
            self.fusion = None
            input_dim = visual_dim
        else:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")

        input_dim += consistency_dim
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            num_classes=len(config["dataset"]["label_mapping"]),
            hidden_dims=[shared_dim],
            dropout=model_config.get("dropout", 0.1),
        )

    def forward(self, text_inputs, image_inputs):
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        pixel_values = image_inputs["pixel_values"]

        if self.fusion_method == "text_only":
            text_features = self.text_encoder(input_ids, attention_mask)
            return self.classifier(text_features)

        if self.fusion_method == "image_only":
            visual_features = self.visual_encoder(pixel_values)
            return self.classifier(visual_features)

        text_features = self.text_encoder(input_ids, attention_mask)
        visual_features = self.visual_encoder(pixel_values)

        if self.fusion_method == "cross_attention":
            text_sequence = self.text_encoder.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            visual_sequence = self.visual_encoder.forward_patches(pixel_values)
            fused_features = self.fusion(
                text_features=text_features,
                image_features=visual_features,
                text_seq=text_sequence,
                image_seq=visual_sequence,
                text_mask=attention_mask.unsqueeze(-1),
            )
        else:
            fused_features = self.fusion(
                text_features=text_features,
                image_features=visual_features,
            )

        if self.configured_consistency:
            text_vector = F.normalize(
                self.text_consistency_projection(text_features.float()),
                p=2,
                dim=-1,
            )
            visual_vector = F.normalize(
                self.visual_consistency_projection(visual_features.float()),
                p=2,
                dim=-1,
            )
            similarity = (text_vector * visual_vector).sum(dim=-1, keepdim=True)
            fused_features = torch.cat([fused_features, similarity], dim=-1)

        return self.classifier(fused_features)
