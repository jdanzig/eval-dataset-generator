"""Render the difficulty proof: target-model failure rate by measured difficulty tier.

Reads build_report.json (written by `python -m app.cli build`) and writes docs/difficulty_proof.png.
The point of the chart: an UNSUPERVISED signal (self-consistency) tracks a SUPERVISED one (accuracy).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report = json.load(open(os.path.join(HERE, "build_report.json")))
proof = report["proof"]

tiers = [t for t in ("easy", "medium", "hard") if t in proof]
rates = [proof[t]["failure_rate"] * 100 for t in tiers]
ns = [proof[t]["n"] for t in tiers]
colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#E53935"}

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
bars = ax.bar(tiers, rates, color=[colors[t] for t in tiers], width=0.6)
for b, r, n in zip(bars, rates, ns):
    ax.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.0f}%\n(n={n})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("target-model failure rate (vs strong-model reference)")
ax.set_xlabel("measured difficulty tier (self-consistency, unsupervised)")
ax.set_title("Self-consistency predicts where the target model fails")
ax.set_ylim(0, max(rates) * 1.25 + 5)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
out = os.path.join(HERE, "docs", "difficulty_proof.png")
fig.savefig(out)
print("wrote", out)
