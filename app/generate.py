"""Orchestrator: turn ingested traffic into a coverage-balanced, versioned eval set.

Flow: cluster real traffic into intents → take a 50-case seed → auto-label (low-confidence to a
review queue) → grow to a target size with paraphrase (medium) + adversarial (hard) variants,
oversampling the thinnest clusters/long-tail first → report coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .augment import adversarials, paraphrases
from .cluster import cluster_intents, cluster_sizes
from .dataset import EvalCase
from .ingest import load_traffic
from .label import ReviewQueue, auto_label


@dataclass
class BuildReport:
    seed_count: int
    final_count: int
    intents: int
    review_queue_size: int
    by_source: dict[str, int] = field(default_factory=dict)


def build_eval_set(traffic_n: int = 500, seed_n: int = 50, target: int = 2000,
                   min_cluster_size: int = 5, traffic: list[dict] | None = None
                   ) -> tuple[list[EvalCase], ReviewQueue, BuildReport]:
    if traffic is None:
        traffic = load_traffic(traffic_n)
    questions = [t["question"] for t in traffic]
    labels, names = cluster_intents(questions, min_cluster_size)
    sizes = cluster_sizes(labels)

    queue = ReviewQueue()
    cases: list[EvalCase] = []

    def make(cid, q, expected, intent, difficulty, source, conf, prov):
        needs = conf < 0.6
        case = EvalCase(cid, q, expected, intent, difficulty, source, round(conf, 3), needs, prov)
        cases.append(case)
        if needs:
            queue.enqueue(cid, q, expected, conf)
        return case

    # 1. seed set: first N real questions, auto-labeled
    for i in range(min(seed_n, len(traffic))):
        t = traffic[i]
        lab = auto_label(t["question"], t["reference"])
        make(f"seed-{i}", t["question"], lab.expected, names[int(labels[i])], "easy",
             "seed", lab.confidence, {"topic": t["topic"], "cluster": int(labels[i])})

    # 2. add the remaining real traffic as more easy cases (the raw coverage base)
    for i in range(seed_n, len(traffic)):
        t = traffic[i]
        lab = auto_label(t["question"], t["reference"])
        make(f"real-{i}", t["question"], lab.expected, names[int(labels[i])], "easy",
             "seed", lab.confidence, {"topic": t["topic"], "cluster": int(labels[i])})

    # 3. augment to target — ROUND-ROBIN across intents so small clusters and the long tail get
    #    equal augmentation turns (oversampled relative to their size), rather than letting the
    #    biggest cluster swallow the budget.
    by_intent: dict[str, list[EvalCase]] = {}
    for c in list(cases):
        by_intent.setdefault(c.intent, []).append(c)
    # each intent gets a queue of (parent, variant_texts) work items
    work: dict[str, list[tuple[EvalCase, str, str, float]]] = {}
    for intent, members in by_intent.items():
        items: list[tuple[EvalCase, str, str, float]] = []
        for parent in members:
            for p in paraphrases(parent.question):
                items.append((parent, p, "medium", parent.confidence * 0.9))
            for a in adversarials(parent.question):
                items.append((parent, a, "hard", parent.confidence * 0.8))
        work[intent] = items

    cursors = {intent: 0 for intent in work}
    vid = 0
    progressing = True
    while len(cases) < target and progressing:
        progressing = False
        for intent in work:  # one item per intent per round
            if len(cases) >= target:
                break
            cur = cursors[intent]
            if cur >= len(work[intent]):
                continue
            parent, text, difficulty, conf = work[intent][cur]
            cursors[intent] += 1
            progressing = True
            src = "paraphrase" if difficulty == "medium" else "adversarial"
            make(f"aug-{vid}", text, parent.expected, intent, difficulty, src, conf,
                 {"parent": parent.id})
            vid += 1

    by_source: dict[str, int] = {}
    for c in cases:
        by_source[c.source] = by_source.get(c.source, 0) + 1

    report = BuildReport(seed_count=min(seed_n, len(traffic)), final_count=len(cases),
                         intents=len([k for k in names if k != -1]),
                         review_queue_size=len(queue.pending()), by_source=by_source)
    return cases, queue, report
