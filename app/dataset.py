"""Eval-case model + versioned JSON persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class EvalCase:
    id: str
    question: str
    intent: str
    source: str                 # traffic | paraphrase | adversarial
    reference: str              # the LLM-proposed label
    confidence: float
    ambiguous: bool
    needs_review: bool
    difficulty: float           # self-consistency difficulty in [0,1]
    tier: str                   # easy | medium | hard (from measured difficulty)
    correct: bool | None = None  # did the target model get it right (for the proof)?


def tier_of(difficulty: float) -> str:
    """Thresholds calibrated on the observed difficulty distribution (roughly terciles)."""
    if difficulty < 0.14:
        return "easy"
    if difficulty < 0.24:
        return "medium"
    return "hard"


def save_dataset(path: str, cases: list[EvalCase], version: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "count": len(cases),
                   "cases": [asdict(c) for c in cases]}, f, indent=1)


def load_dataset(path: str) -> tuple[str, list[EvalCase]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["version"], [EvalCase(**c) for c in d["cases"]]
