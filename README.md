# Automated Eval Dataset Generator

Turns raw production traffic into a **labeled, deduplicated, coverage-balanced eval set** —
growing from a ~50-case seed to 2,000+ cases while mining the long tail for edge cases.

> 🎯 **Project purpose:** *"Automated eval-set generation from production traffic, growing
> coverage from 50 to 2,000+ labeled cases and catching edge cases pre-release."*

Uses a real public dataset (SQuAD) as stand-in traffic. Runs **fully offline** (local embeddings,
HDBSCAN, deterministic augmentation) — no API keys.

---

## Pipeline

```
public dataset (SQuAD) as traffic
   │  (fetched from spread-out offsets → diverse topics)
   ▼
embed (local hashing) → HDBSCAN clustering → intents  (+ noise = long-tail edge cases)
   │
   ▼
50-case seed → auto-label with confidence → low-confidence cases → review queue
   │
   ▼
grow to target via paraphrase (medium) + adversarial (hard) variants,
   ROUND-ROBIN across intents so small clusters / long-tail get equal turns
   │
   ▼
coverage heatmap (intent × difficulty) + thin-cell gaps + versioned eval_set.json
```

## Quick start

```bash
make install
make test        # 5 unit tests (offline, synthetic traffic)
make build       # grow 50 → 2,000+, write eval_set.json + coverage.html
make run         # review-queue + coverage API on :8000
```

`make build` output (abridged):

```
grew eval set: 50 seed → 2000 labeled cases
intents discovered: 6  (+ long-tail bucket)
by source: {'seed': 500, 'paraphrase': 752, 'adversarial': 748}
review queue (low-confidence): 293 cases

coverage heatmap (intent × difficulty):
intent                |    easy  medium    hard |   total
long-tail             |     427     460     456 |    1343
bowl/super/50         |      43     172     172 |     387
dutch/revolt/when     |       6      24      24 |      54
...
thin cells still needing coverage (<3): 0
```

Open `coverage.html` for the color-coded heatmap.

## Why each choice

- **HDBSCAN, not k-means:** finds the natural number of intents and, crucially, marks
  unclustered questions as noise (label -1). That noise bucket is the **long tail** where the
  most valuable edge cases live, so we keep it as a first-class intent.
- **Confidence + review queue:** we *have* gold references, so auto-labels start confident;
  points are docked for essay-length or vague/ungrounded answers, and anything below threshold
  (or with no reference) is routed to a human review queue instead of trusted blindly.
- **Round-robin augmentation:** naive augmentation explodes the biggest cluster. Giving each
  intent one turn per round oversamples small clusters and the long tail relative to their size —
  which is what actually fills coverage gaps (thin cells → 0).
- **Difficulty bands:** seed/real = easy, paraphrase = medium, adversarial = hard, so the heatmap
  shows coverage across *difficulty*, not just topic.

## API

| Route | Purpose |
|---|---|
| `GET /stats` | seed→final counts, intents, source breakdown, queue size |
| `GET /coverage?fmt=text` | intent×difficulty heatmap + thin cells |
| `GET /review/pending` | low-confidence cases awaiting review |
| `POST /review/{id}` | resolve: `{"decision":"confirmed\|corrected\|rejected","corrected":"..."}` |

## Honest notes

- Traffic is a public dataset, not real logs — swap `app/ingest.py` for your log source.
- The offline augmenter uses deterministic mutations (paraphrase templates, light corruption for
  adversarial). An LLM augmenter would produce more natural variants — same interface.
- HDBSCAN routes a large share of diverse questions to the long-tail bucket; that's intentional
  (it's the edge-case reservoir), not a clustering failure.

## Layout

```
app/  ingest · embed · cluster (HDBSCAN) · label (+review queue) · augment · coverage · generate · api · cli
tests/ label · augment · generate · coverage
```
