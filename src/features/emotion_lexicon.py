import os
import numpy as np
from collections import Counter

# Ordered list of the 10 NRC emotions (including polarity)
EMOTIONS = [
    "anger",
    "fear",
    "anticipation",
    "trust",
    "surprise",
    "sadness",
    "joy",
    "disgust",
    "positive",
    "negative",
]

def _load_nrc_lexicon(path: str) -> dict:
    """Load the NRC Emotion Lexicon from *path*.
    Returns a mapping ``word -> set(emotions)``.
    """
    lex = {}
    if not os.path.isfile(path):
        # Gracefully handle missing lexicon file – return empty mapping.
        return lex
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emo, flag = parts
            if int(flag) == 1:
                lex.setdefault(word, set()).add(emo)
    return lex

# Expected location of the lexicon file relative to repository root.
_lexicon_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nrc_emotion_lexicon.txt")
_NRC_LEXICON = _load_nrc_lexicon(_lexicon_path)

def nrc_emotion_vector(text: str) -> np.ndarray:
    """Return a normalized 10‑dim vector for *text*.
    Vector follows the order defined in :data:`EMOTIONS`.
    """
    tokens = [t.lower() for t in text.split()]
    counts = Counter()
    for tok in tokens:
        if tok in _NRC_LEXICON:
            for emo in _NRC_LEXICON[tok]:
                counts[emo] += 1
    vec = np.array([counts.get(e, 0) for e in EMOTIONS], dtype=np.float32)
    total = max(len(tokens), 1)
    return vec / total
