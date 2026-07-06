"""Orchestrator: turn synthesized production traffic into a difficulty-scored, versioned eval set.

Flow:
  1. synthesize realistic growth-analytics traffic (LLM, deduped)               [traffic.py]
  2. cluster it into intents (HDBSCAN over semantic embeddings)                 [cluster.py]
  3. auto-label each case (strong model proposes a reference + confidence;
     low-confidence / ambiguous cases route to a human-review queue)            [label.py]
  4. score DIFFICULTY by the target model's self-consistency (unsupervised)     [difficulty.py]
     and, for the proof, whether the target actually got it right vs reference
  5. fill the thinnest intents with genuine LLM variants                        [augment.py]

The headline is step 4: an unsupervised difficulty signal that predicts where the target model
fails. `BuildReport.proof` bins cases by measured difficulty tier and reports the failure rate in
each — if easy-tier < hard-tier, the signal is real.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .augment import novel, variants
from .cluster import cluster_intents, cluster_sizes
from .dataset import EvalCase, tier_of
from .difficulty import STRONG, TARGET, is_correct, sample_target, self_consistency_difficulty
from .label import ReviewQueue, auto_label
from .llm import complete
from .traffic import synthesize_traffic


@dataclass
class BuildReport:
    traffic_count: int
    final_count: int
    intents: int
    review_queue_size: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_tier: dict[str, int] = field(default_factory=dict)
    proof: dict = field(default_factory=dict)   # tier -> {n, failure_rate}


def _score_case(cid: str, question: str, intent: str, source: str, queue: ReviewQueue,
                k: int) -> EvalCase:
    lab = auto_label(question)
    samples = sample_target(question, k=k)
    diff = self_consistency_difficulty(samples)
    # a clean deterministic answer for the correctness check (the proof)
    det = complete(TARGET, question, max_tokens=200, temperature=0.0)
    correct = is_correct(det, lab.answer, question) if lab.answer else None
    case = EvalCase(cid, question, intent, source, lab.answer, lab.confidence, lab.ambiguous,
                    lab.needs_review, diff, tier_of(diff), correct)
    if lab.needs_review:
        queue.enqueue(cid, question, lab.answer, lab.confidence, lab.ambiguous)
    return case


def _proof(cases: list[EvalCase]) -> dict:
    out: dict[str, dict] = {}
    for tier in ("easy", "medium", "hard"):
        graded = [c for c in cases if c.tier == tier and c.correct is not None]
        if not graded:
            continue
        fails = sum(1 for c in graded if not c.correct)
        out[tier] = {"n": len(graded), "failures": fails,
                     "failure_rate": round(fails / len(graded), 3)}
    return out


def build_eval_set(traffic_n: int = 60, min_cluster_size: int = 3, k: int = 4,
                   augment_intents: int = 4, questions: list[str] | None = None
                   ) -> tuple[list[EvalCase], ReviewQueue, BuildReport]:
    if questions is None:
        questions = synthesize_traffic(traffic_n)
    questions = questions[:traffic_n]
    labels, names = cluster_intents(questions, min_cluster_size)

    queue = ReviewQueue()
    cases: list[EvalCase] = []
    for i, q in enumerate(questions):
        intent = names[int(labels[i])]
        cases.append(_score_case(f"mined-{i}", q, intent, "traffic", queue, k))

    # fill the thinnest real intents with genuine LLM variants
    sizes = cluster_sizes(labels)
    real = [(cid, names[cid]) for cid in sizes if cid != -1]
    thin = sorted(real, key=lambda x: sizes[x[0]])[:augment_intents]
    vid = 0
    for cid, intent in thin:
        seed = next(c for c in cases if c.intent == intent)
        v = variants(seed.question, intent)
        for src, cands in (("paraphrase", v["paraphrase"]), ("adversarial", v["adversarial"])):
            for cand in novel(cands, [c.question for c in cases]):
                cases.append(_score_case(f"aug-{vid}", cand, intent, src, queue, k))
                vid += 1

    by_source: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for c in cases:
        by_source[c.source] = by_source.get(c.source, 0) + 1
        by_tier[c.tier] = by_tier.get(c.tier, 0) + 1

    report = BuildReport(
        traffic_count=len(questions), final_count=len(cases),
        intents=len([k2 for k2 in names if k2 != -1]),
        review_queue_size=len(queue.pending()), by_source=by_source, by_tier=by_tier,
        proof=_proof(cases))
    return cases, queue, report
