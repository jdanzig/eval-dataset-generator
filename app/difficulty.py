"""Difficulty scoring by the TARGET model's own uncertainty (self-consistency).

We sample the target model K times at temperature and measure how much its answers DISAGREE
(embedding dispersion). High disagreement = the model is uncertain = a hard case. This is an
UNSUPERVISED signal — it needs no gold answer.

The proof it's meaningful (non-circular): self-consistency PREDICTS accuracy. We separately score
each case's correctness (target's answer vs the strong-model reference, judged) and show that the
high-difficulty cases have a far higher failure rate than the low-difficulty ones. An unsupervised
signal predicting a supervised one is a real property of the case, not a definition.
"""
from __future__ import annotations

import os

from .llm import complete

TARGET = os.getenv("EVALGEN_TARGET_MODEL", "claude-haiku-4-5-20251001")
STRONG = os.getenv("EVALGEN_STRONG_MODEL", "claude-opus-4-8")


def sample_target(question: str, k: int = 4) -> list[str]:
    return [complete(TARGET, question, max_tokens=200, temperature=1.0) for _ in range(k)]


def self_consistency_difficulty(answers: list[str]) -> float:
    """1 - mean pairwise cosine similarity of the sampled answers, in [0,1]. Higher = more
    disagreement = harder."""
    from .embed import embed
    import numpy as np

    valid = [a for a in answers if a.strip()]
    if len(valid) < 2:
        return 0.0
    v = embed(valid)
    sims = v @ v.T
    n = len(valid)
    off = (sims.sum() - n) / (n * n - n)   # mean of off-diagonal cosine similarities
    return round(float(max(0.0, min(1.0, 1.0 - off))), 3)


def is_correct(target_answer: str, reference: str, question: str) -> bool:
    """Judge (strong model): does the target's answer match the reference / correctly answer it?"""
    out = complete(
        STRONG,
        "Does the candidate answer correctly answer the question, consistent with the reference? "
        "Reply only 'yes' or 'no'.\n\n"
        f"Question: {question}\nReference: {reference}\nCandidate: {target_answer}",
        max_tokens=4)
    return out.strip().lower().startswith("y")
