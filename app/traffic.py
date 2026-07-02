"""Synthesize realistic 'production traffic' for a growth-analytics assistant.

Real traffic is messy: precise experts and confused beginners, one-liners and rambles, typos,
ambiguity. We generate a diverse pool across growth sub-topics and personas (via the model), then
dedup near-duplicates by embedding similarity — so the mining, clustering, and difficulty scoring
downstream have something real to work on. Cached to disk after first generation.
"""
from __future__ import annotations

import json
import os
import re

from .llm import complete

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, ".traffic_cache.json")
MODEL = os.getenv("EVALGEN_STRONG_MODEL", "claude-opus-4-8")

# (topic, persona/style) pairs steer diversity without us hand-writing questions
BATCHES = [
    ("experimentation statistics (A/B tests, significance, variance reduction, peeking)",
     "a precise senior data scientist"),
    ("experimentation statistics (bandits, holdouts, sample ratio mismatch, interference)",
     "a confused PM who half-remembers the terms"),
    ("growth metrics (activation, retention, churn, dormancy, cohorts)",
     "a founder who mixes up similar metrics"),
    ("growth metrics (LTV, CAC, payback, k-factor, north star)",
     "a terse analyst who writes short fragments"),
    ("analytics instrumentation (event taxonomy, identity resolution, sessionization, attribution)",
     "an engineer asking implementation questions"),
    ("analytics instrumentation (bot filtering, dedup, client vs server tracking, data quality)",
     "someone who writes long rambling questions with typos"),
    ("growth systems (PLG, referral loops, onboarding/activation, paywalls, pricing)",
     "a growth marketer, casual tone"),
    ("edge cases that are ambiguous or sit between two concepts",
     "a beginner unsure what to even ask"),
]


def _parse_questions(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        if len(line) > 8 and "?" in line or (len(line) > 12 and line.endswith((".", ""))):
            out.append(line)
    return out


def synthesize_traffic(n: int = 120, per_batch: int = 16) -> list[str]:
    if os.path.exists(CACHE):
        data = json.load(open(CACHE, encoding="utf-8"))
        if len(data) >= n:
            return data[:n]

    questions: list[str] = []
    for topic, persona in BATCHES:
        text = complete(
            MODEL,
            f"Generate {per_batch} realistic questions that {persona} might type into a "
            f"growth-analytics assistant, about {topic}. Make them varied in length and phrasing, "
            f"like real user traffic — some sloppy, some precise, some ambiguous. One question per "
            f"line, no numbering, no preamble.",
            temperature=1.0, max_tokens=900)
        questions.extend(_parse_questions(text))

    # dedup near-duplicates by embedding similarity
    questions = _dedup(questions)
    json.dump(questions, open(CACHE, "w", encoding="utf-8"))
    return questions[:n]


def _dedup(questions: list[str], threshold: float = 0.92) -> list[str]:
    from .embed import embed
    import numpy as np

    if not questions:
        return []
    vecs = embed(questions)
    keep, kept_vecs = [], []
    for q, v in zip(questions, vecs):
        if kept_vecs and float(np.max(np.asarray(kept_vecs) @ v)) >= threshold:
            continue
        keep.append(q); kept_vecs.append(v)
    return keep
