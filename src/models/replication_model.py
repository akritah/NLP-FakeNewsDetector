import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .emotion_features import EmotionFeatureExtractor
from .fusion import MultimodalFusion
from .vae import VAE
from .classifier import MLPClassifier

class ReplicationModel(nn.Module):
    """
    The complete faithful replication of the multimodal pipeline:
    RoBERTa (Semantic) + DistilBERT/NRC (Emotion) + ViT (Visual) -> VAE -> MLP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Semantic Branch
        self.text_encoder = TextEncoder(
            checkpoint=config['model']['text_encoder']['checkpoint'],
            freeze=config['model']['text_encoder']['freeze'],
            pooling=config['model']['text_encoder']['pooling']
        )

        # 2. Emotion Branch
        self.emotion_encoder = EmotionFeatureExtractor(
            checkpoint=config['model']['emotion_encoder']['checkpoint'],
            freeze=config['model']['emotion_encoder']['freeze']
        )

        # 3. Visual Branch
        self.image_encoder = ImageEncoder(
            checkpoint=config['model']['image_encoder']['checkpoint'],
            freeze=config['model']['image_encoder']['freeze']
        )

        # 4. Fusion (Concatenation)
        text_dim = self.text_encoder.roberta.config.hidden_size
        image_dim = self.image_encoder.vit.config.hidden_size
        # Emotion dim = DistilBERT(768) + NRC(8) = 776
        emotion_dim = 768 + 8

        self.fusion = MultimodalFusion(
            text_dim=text_dim,
            image_dim=image_dim,
            fusion_method="concat",
            shared_dim=config['model']['hidden_dim']
        )

        # Determine concatenated dimension for VAE input
        # text_proj(shared_dim) + image_proj(shared_dim) + emotion_dim
        fused_dim = (config['model']['hidden_dim'] * 2) + emotion_dim

        # 5. VAE Compression
        self.vae = VAE(
            input_dim=fused_dim,
            latent_dim=config['model']['vae_latent_dim']
        )

        # 6. Classifier
        # Input is the latent dimension of the VAE
        self.classifier = MLPClassifier(
            input_dim=config['model']['vae_latent_dim'],
            num_classes=len(config['dataset']['label_mapping']),
            hidden_dims=[config['model']['hidden_dim']],
            dropout=config['model']['dropout']
        )

    def forward(self, text_inputs, image_inputs, raw_text_batch):
        # Extract Semantic features
        text_features = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])

        # Extract Emotion features
        emotion_features = self.emotion_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
            raw_text_batch
        )

        # Extract Visual features
        image_features = self.image_encoder(image_inputs['pixel_values'])

        # Fusion (Concatenation)
        fused_vector = self.fusion(
            text_features=text_features,
            image_features=image_features,
            emotion_features=emotion_features
        )

        # VAE Compression
        # We use the latent mean (mu) as the representation for classification
        recon, mu, logvar = self.vae(fused_vector)

        # Classification
        logits = self.classifier(mu)

        return logits, recon, mu, logvar
