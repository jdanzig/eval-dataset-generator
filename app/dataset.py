"""Eval-case model + versioned JSON persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str
    expected: str
    intent: str
    difficulty: str          # easy | medium | hard
    source: str              # seed | paraphrase | adversarial
    confidence: float
    needs_review: bool = False
    provenance: dict = field(default_factory=dict)


def save_dataset(path: str, cases: list[EvalCase], version: str) -> None:
    payload = {"version": version, "count": len(cases), "cases": [asdict(c) for c in cases]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def load_dataset(path: str) -> tuple[str, list[EvalCase]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["version"], [EvalCase(**c) for c in payload["cases"]]
