import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Optional

class TextEncoder(nn.Module):
    """
    Text encoder using a pretrained RoBERTa model.
    """
    def __init__(
        self,
        checkpoint: str = "roberta-base",
        freeze: bool = True,
        pooling: str = "cls"
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(checkpoint)
        self.pooling = pooling

        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state: [batch, seq_len, hidden_dim]
        # pooler_output: [batch, hidden_dim] (The CLS token processed by a linear layer)

        if self.pooling == "cls":
            # Use the first token (CLS) representation directly from last_hidden_state
            # for more consistency across different RoBERTa variants
            return outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Average pooling across the sequence length
            # Masking padding tokens
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
