import torch
from transformers import DistilBertModel, DistilBertTokenizer

# Lazy-loaded global variables
_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is None:
        _model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        _model.eval()
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def distilbert_emotion_embedding(text: str) -> torch.Tensor:
    """Return a 768‑dim emotion embedding for *text*.
    
    The function tokenizes the input, passes it through DistilBERT, and extracts the hidden state
    of the first token (CLS) as a fixed‑size representation.
    Runs with torch.no_grad() for efficiency.
    """
    if not isinstance(text, str):
        raise ValueError('Input must be a string')
    _load_model()
    inputs = _tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = _model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0)
