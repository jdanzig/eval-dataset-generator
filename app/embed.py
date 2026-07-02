"""Zero-dependency local embeddings (hashing trick over content tokens) for clustering."""
from __future__ import annotations

import hashlib
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "what", "which", "who", "how",
         "did", "does", "was", "were", "on", "for", "and", "or", "at", "by", "from"}


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def _h(tok: str, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big") % dim


def embed(texts: list[str], dim: int = 512) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        toks = tokens(text)
        for t in toks:
            out[i, _h(t, dim)] += 1.0
        for a, b in zip(toks, toks[1:]):
            out[i, _h(a + "_" + b, dim)] += 1.0
        nrm = np.linalg.norm(out[i])
        if nrm > 0:
            out[i] /= nrm
    return out
