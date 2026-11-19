import torch
import torch.nn as nn
import os
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Tuple, Dict, List
import numpy as np

class EmotionFeatureExtractor(nn.Module):
    """
    Extracts emotion-related features from text using a hybrid approach:
    1. Deep Learning: DistilBERT contextual embeddings.
    2. Lexical: NRC Emotion Lexicon-based counts.
    """
    def __init__(
        self,
        checkpoint: str = "distilbert-base-uncased",
        freeze: bool = True,
        nrc_lexicon_path: str = None
    ):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(checkpoint)
        self.tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)

        if freeze:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        self.emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
        # NRC Lexicon loading
        self.emotion_lexicon = self._load_nrc_lexicon(nrc_lexicon_path)

    def _load_nrc_lexicon(self, path: str) -> Dict[str, List[float]]:
        """
        Loads the NRC Emotion Lexicon.
        If no path is provided, initializes an empty lexicon (place-holder for reproduction).
        """
        lexicon = {}
        if path and os.path.exists(path):
            label_index = {label: index for index, label in enumerate(self.emotion_labels)}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        word, emotion, flag = parts
                        if emotion in label_index and flag == "1":
                            scores = lexicon.setdefault(
                                word.lower(), [0.0] * len(self.emotion_labels)
                            )
                            scores[label_index[emotion]] = 1.0
                    elif len(parts) >= 9:
                        word = parts[0].lower()
                        raw_scores = parts[1:1 + len(self.emotion_labels)]
                        lexicon[word] = [
                            float(value) if value != '?' else 0.0
                            for value in raw_scores
                        ]
        return lexicon

    def extract_lexical_emotions(self, text: str) -> torch.Tensor:
        """
        Calculates emotion scores based on the NRC Lexicon.
        """
        words = text.lower().split()
        counts = np.zeros(len(self.emotion_labels))

        if not self.emotion_lexicon:
            return torch.tensor(counts, dtype=torch.float32)

        for word in words:
            if word in self.emotion_lexicon:
                counts += np.array(self.emotion_lexicon[word])

        # Normalize by text length to avoid bias toward longer posts
        if len(words) > 0:
            counts = counts / len(words)

        return torch.tensor(counts, dtype=torch.float32)

    def forward(self, input_ids, attention_mask, raw_text_batch: List[str]):
        """
        Returns a fused emotion representation.
        """
        # 1. Learned features from DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token as the emotion-representative embedding
        learned_emotion = outputs.last_hidden_state[:, 0, :] # [batch, 768]

        # 2. Lexical features from NRC
        lexical_emotions = []
        for text in raw_text_batch:
            lexical_emotions.append(self.extract_lexical_emotions(text))

        lexical_tensor = torch.stack(lexical_emotions).to(learned_emotion.device) # [batch, 8]

        # Concatenate learned and lexical features
        return torch.cat([learned_emotion, lexical_tensor], dim=-1) # [batch, 768 + 8]
