# -*- coding: utf-8 -*-
"""Generate Exp2_L analysis plots."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

LOG_DIR = "E:/PY_Project/LDA/logs/sweep/20260508_164213_Exp2_L"

records = []
for fname in os.listdir(LOG_DIR):
    if not fname.endswith("_metrics.json"):
        continue
    parts = fname.split("_")
    algo = parts[0]
    l_mean = float(parts[2].replace("mean", ""))
    seed = int(parts[3][1:])
    path = os.path.join(LOG_DIR, fname)
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    records.append({
        "algo": algo, "L_mean": l_mean, "seed": seed,
        "paoi_mean": s["paoi_mean"], "q_mean": s["q_mean"],
        "e_bs_mean": s["e_bs_mean"], "e_sat_mean": s["e_sat_mean"],
    })

L_values = sorted(set(r["L_mean"] for r in records))
L_labels = [f"{v/1e6:.0f}" for v in L_values]
algos = ["AC", "COB", "LDA", "MTD"]
colors = {"LDA": "#e74c3c", "AC": "#3498db", "COB": "#95a5a6", "MTD": "#2ecc71"}
markers = {"LDA": "o", "AC": "s", "COB": "^", "MTD": "D"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Exp2_L: Impact of Task Load L_mean on System Performance", fontsize=15, fontweight='bold')

metrics_plot = [
    ("paoi_mean", "PAoI", axes[0, 0]),
    ("q_mean", "Queue Backlog (bits)", axes[0, 1]),
    ("e_bs_mean", "BS Energy (J)", axes[1, 0]),
    ("e_sat_mean", "Satellite Energy (J)", axes[1, 1]),
]

for mkey, mname, ax in metrics_plot:
    for algo in algos:
        means, stds = [], []
        for L in L_values:
            vals = [r[mkey] for r in records if r["algo"] == algo and r["L_mean"] == L]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        ax.errorbar(range(len(L_values)), means, yerr=stds,
                    color=colors[algo], marker=markers[algo], capsize=5,
                    capthick=1.5, linewidth=2, markersize=8, label=algo)
    ax.set_xticks(range(len(L_values)))
    ax.set_xticklabels(L_labels)
    ax.set_xlabel("L_mean (Mbit)", fontsize=11)
    ax.set_ylabel(mname, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# Add energy budget lines
axes[1, 0].axhline(y=180, color='red', linestyle='--', linewidth=1, alpha=0.5, label='E_BS_max=180J')
axes[1, 1].axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='E_Sat_max=80J')

plt.tight_layout()
save_path = "E:/PY_Project/LDA/results/exp2_L_analysis.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"Plot saved to: {save_path}")

# ---- Extra: LDA vs AC detailed comparison ----
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Ablation: LDA vs AC (PAoI Term Contribution)", fontsize=14, fontweight='bold')

# PAoI comparison
ax = axes2[0]
for algo in ["LDA", "AC"]:
    means = [np.mean([r["paoi_mean"] for r in records if r["algo"] == algo and r["L_mean"] == L]) for L in L_values]
    ax.plot(L_labels, means, color=colors[algo], marker=markers[algo], linewidth=2, markersize=8, label=algo)
ax.set_xlabel("L_mean (Mbit)")
ax.set_ylabel("PAoI")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("PAoI Comparison")

# Gap
ax2 = axes2[1]
gaps, gap_pcts = [], []
for L in L_values:
    lda_m = np.mean([r["paoi_mean"] for r in records if r["algo"] == "LDA" and r["L_mean"] == L])
    ac_m = np.mean([r["paoi_mean"] for r in records if r["algo"] == "AC" and r["L_mean"] == L])
    gaps.append(lda_m - ac_m)
    gap_pcts.append((lda_m - ac_m) / ac_m * 100)
bars = ax2.bar(L_labels, gap_pcts, color=['green' if g < 0 else 'red' for g in gap_pcts], alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_xlabel("L_mean (Mbit)")
ax2.set_ylabel("LDA PAoI - AC PAoI (%)")
ax2.set_title("PAoI Gap (negative = LDA better)")
for bar, pct in zip(bars, gap_pcts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if bar.get_height() >= 0 else -1.2),
             f'{pct:+.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
save_path2 = "E:/PY_Project/LDA/results/exp2_L_ablation.png"
plt.savefig(save_path2, dpi=200, bbox_inches='tight')
print(f"Ablation plot saved to: {save_path2}")
