"""LLM augmentation — genuine variants, not string templates.

From a seed question the model writes a *paraphrase* (same intent, different phrasing — tests
robustness) and an *adversarial* variant (a nearby-but-harder case in the same intent — tricky
edge, easy-to-confuse sibling concept). These fill thin clusters with realistic cases instead of
mechanically-mangled ones. Variants are deduped against the existing pool by embedding similarity
so augmentation never just restates a question already present.
"""
from __future__ import annotations

import os
import re

from .llm import complete

MODEL = os.getenv("EVALGEN_STRONG_MODEL", "claude-opus-4-8")


def _lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        # drop preamble lines the model adds despite "no preamble" ("Here are 2 harder questions:")
        if re.match(r"(?i)^(here (are|is)|sure|okay|certainly)\b", line):
            continue
        if len(line) > 8:
            out.append(line)
    return out


def variants(question: str, intent: str) -> dict[str, list[str]]:
    """Return {'paraphrase': [...], 'adversarial': [...]} of genuine LLM variants for one seed."""
    para = complete(
        MODEL,
        f"Rewrite this growth-analytics question 2 times, same meaning, different phrasing and "
        f"tone (as a real user might ask it). One per line, no numbering.\n\n{question}",
        temperature=1.0, max_tokens=200)
    adv = complete(
        MODEL,
        f"This question is about '{intent}'. Write 2 HARDER questions in the same area that a model "
        f"is likely to get subtly wrong — edge cases, or easy-to-confuse sibling concepts. One per "
        f"line, no numbering.\n\n{question}",
        temperature=1.0, max_tokens=220)
    return {"paraphrase": _lines(para)[:2], "adversarial": _lines(adv)[:2]}


def novel(candidates: list[str], existing: list[str], threshold: float = 0.9) -> list[str]:
    """Keep only candidates not near-duplicate of anything in `existing`."""
    from .embed import embed
    import numpy as np

    if not candidates:
        return []
    ex = embed(existing) if existing else None
    cand = embed(candidates)
    keep = []
    for q, v in zip(candidates, cand):
        if ex is not None and float(np.max(ex @ v)) >= threshold:
            continue
        keep.append(q)
    return keep
