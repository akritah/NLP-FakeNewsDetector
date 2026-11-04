import unittest
import torch
import numpy as np
from src.models.text_encoder import TextEncoder
from src.models.image_encoder import ImageEncoder
from src.models.fusion import MultimodalFusion
from src.models.classifier import MLPClassifier
from src.models.multimodal_model import MultimodalModel

class TestBaselineComponents(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.text_dim = 768
        self.image_dim = 768
        self.shared_dim = 256

    def test_text_encoder_dims(self):
        encoder = TextEncoder(pooling="cls")
        ids = torch.randint(0, 100, (self.batch_size, 128))
        mask = torch.ones((self.batch_size, 128))
        out = encoder(ids, mask)
        self.assertEqual(out.shape, (self.batch_size, self.text_dim))

    def test_image_encoder_dims(self):
        encoder = ImageEncoder()
        pixels = torch.randn((self.batch_size, 3, 224, 224))
        out = encoder(pixels)
        self.assertEqual(out.shape, (self.batch_size, self.image_dim))

        # Test patch extraction for cross-attn
        patches = encoder.forward_patches(pixels)
        self.assertEqual(patches.ndim, 3)
        self.assertEqual(patches.shape[-1], self.image_dim)

    def test_fusion_concat(self):
        fusion = MultimodalFusion(self.text_dim, self.image_dim, fusion_method="concat", shared_dim=self.shared_dim)
        t = torch.randn((self.batch_size, self.text_dim))
        i = torch.randn((self.batch_size, self.image_dim))
        out = fusion(t, i)
        # concat: shared_dim * 2
        self.assertEqual(out.shape, (self.batch_size, self.shared_dim * 2))

    def test_fusion_cross_attention(self):
        fusion = MultimodalFusion(self.text_dim, self.image_dim, fusion_method="cross_attention", shared_dim=self.shared_dim)
        t_pooled = torch.randn((self.batch_size, self.text_dim))
        i_pooled = torch.randn((self.batch_size, self.image_dim))
        t_seq = torch.randn((self.batch_size, 128, self.text_dim))
        i_seq = torch.randn((self.batch_size, 197, self.image_dim))

        out = fusion(t_pooled, i_pooled, text_seq=t_seq, image_seq=i_seq)
        # cross_attn: shared_dim * 2 (from concatenated pooled fused vectors)
        self.assertEqual(out.shape, (self.batch_size, self.shared_dim * 2))

    def test_classifier_output(self):
        classifier = MLPClassifier(input_dim=512, num_classes=2)
        x = torch.randn((self.batch_size, 512))
        out = classifier(x)
        self.assertEqual(out.shape, (self.batch_size, 2))

if __name__ == "__main__":
    unittest.main()
