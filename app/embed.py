"""Real semantic embeddings via a local sentence-transformers model (no API cost). Replaces the
hash-trick embedder so HDBSCAN finds real intent clusters instead of lexical noise. `tokens` is a
small content-word tokenizer used only to name clusters."""
from __future__ import annotations
import re
import numpy as np

_MODEL = None
_NAME = "all-MiniLM-L6-v2"
_STOP = {"the","a","an","of","to","in","is","are","what","how","does","do","and","or","for","on",
         "with","when","why","which","between","vs","my","we","our","i","you","should","can","if",
         "that","this","it","be","as","at","by","from","about","get","got"}

def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_NAME)
    return _MODEL

def embed(texts: list[str]) -> np.ndarray:
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

def tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 2]
