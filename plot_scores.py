import json
import numpy as np
import matplotlib.pyplot as plt

with open("results/scoring_log.json") as f:
    data = json.load(f)

ref_lt = ref_eq = ref_gt = 0
enr_lt = enr_eq = enr_gt = 0
total = 0

for i, item in enumerate(data):
    base = item.get("base_output_eval")
    enr = item.get("output_eval_enriched")
    ref = item.get("output_eval_refined")

    # Choose baseline: base if available, else enriched
    if base:
        base_score = np.mean([base.get("coherence", 0), base.get("creativity", 0), base.get("engagement", 0)])
    elif enr:
        base_score = np.mean([enr.get("coherence", 0), enr.get("creativity", 0), enr.get("engagement", 0)])
    else:
        print(f"[!] Skipping prompt {i} — no base or enriched output")
        continue

    base_score = round(base_score, 2)
    total += 1

    if ref:
        ref_score = round(np.mean([ref.get("coherence", 0), ref.get("creativity", 0), ref.get("engagement", 0)]), 2)
        if ref_score < base_score:
            ref_lt += 1
        elif ref_score == base_score:
            ref_eq += 1
        else:
            ref_gt += 1

    if enr:
        enr_score = round(np.mean([enr.get("coherence", 0), enr.get("creativity", 0), enr.get("engagement", 0)]), 2)
        if enr_score < base_score:
            enr_lt += 1
        elif enr_score == base_score:
            enr_eq += 1
        else:
            enr_gt += 1

# Plot
labels = [
    "Refined < Baseline", "Refined = Baseline", "Refined > Baseline",
    "Enriched < Baseline", "Enriched = Baseline", "Enriched > Baseline"
]
counts = [ref_lt, ref_eq, ref_gt, enr_lt, enr_eq, enr_gt]
colors = ['red', 'gray', 'green', 'orange', 'lightblue', 'darkblue']

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, counts, color=colors)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, count + 0.2, str(count), ha='center', va='bottom')

plt.title(f"Comparison of Refined & Enriched Output Scores vs Baseline — {total} Prompts Compared")
plt.ylabel("Number of Prompts")
plt.ylim(0, max(counts) + 2)
plt.tight_layout()
plt.show()
