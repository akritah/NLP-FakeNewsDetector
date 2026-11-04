import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .fusion import MultimodalFusion
from .classifier import MLPClassifier

class MultimodalModel(nn.Module):
    """
    The complete Multimodal Baseline Model.
    """
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config

        # Encoders
        self.text_encoder = TextEncoder(
            checkpoint=config['model']['text_encoder']['checkpoint'],
            freeze=config['model']['text_encoder']['freeze'],
            pooling=config['model']['text_encoder']['pooling']
        )
        self.image_encoder = ImageEncoder(
            checkpoint=config['model']['image_encoder']['checkpoint'],
            freeze=config['model']['image_encoder']['freeze']
        )

        # Dimensions
        text_dim = self.text_encoder.roberta.config.hidden_size
        image_dim = self.image_encoder.vit.config.hidden_size

        self.fusion_method = config['model']['fusion_method']
        self.fusion = None
        if self.fusion_method in {"concat", "cross_attention"}:
            self.fusion = MultimodalFusion(
                text_dim=text_dim,
                image_dim=image_dim,
                fusion_method=self.fusion_method,
                shared_dim=config['model']['hidden_dim']
            )

        # Determine MLP input dimension
        if self.fusion_method in {"concat", "cross_attention"}:
            # if concat: text_proj(shared_dim) + image_proj(shared_dim)
            mlp_input_dim = config['model']['hidden_dim'] * 2
        elif self.fusion_method == "cross_attention":
            # if cross_attn: t_global(shared_dim) + i_global(shared_dim)
            mlp_input_dim = config['model']['hidden_dim'] * 2
        else:
            # Handle text-only or image-only experiments
            # These will be handled by specific model wrappers for clarity
            mlp_input_dim = text_dim if self.fusion_method == "text_only" else image_dim

        self.classifier = MLPClassifier(
            input_dim=mlp_input_dim,
            num_classes=len(config['dataset']['label_mapping']),
            hidden_dims=[config['model']['hidden_dim']],
            dropout=config['model']['dropout']
        )

    def forward(self, text_inputs, image_inputs):
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']

        if self.fusion_method == "text_only":
            fused_features = self.text_encoder(input_ids, attention_mask)
        elif self.fusion_method == "image_only":
            fused_features = self.image_encoder(image_inputs['pixel_values'])
        else:
            text_seq = self.text_encoder.roberta(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            text_pooled = self.text_encoder(input_ids, attention_mask)
            pixel_values = image_inputs['pixel_values']
            image_seq = self.image_encoder.forward_patches(pixel_values)
            image_pooled = self.image_encoder(pixel_values)
            fused_features = self.fusion(
                text_features=text_pooled,
                image_features=image_pooled,
                text_seq=text_seq,
                image_seq=image_seq,
                text_mask=attention_mask.unsqueeze(-1)
            )

        # Classification
        logits = self.classifier(fused_features)
        return logits
