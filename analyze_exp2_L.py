# -*- coding: utf-8 -*-
"""Analyze Exp2_L sweep results: varying L_mean (task size)."""
import json
import os
import sys
import numpy as np
from collections import defaultdict

# Force UTF-8 output
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
    summary = data["summary"]
    records.append({
        "algo": algo, "L_mean": l_mean, "seed": seed,
        "paoi_mean": summary["paoi_mean"],
        "q_mean": summary["q_mean"],
        "e_bs_mean": summary["e_bs_mean"],
        "e_sat_mean": summary["e_sat_mean"],
        "max_e_queue": summary.get("max_e_queue", np.nan),
        "final_e_queue": summary.get("final_e_queue", np.nan),
    })

L_values = sorted(set(r["L_mean"] for r in records))
algos = sorted(set(r["algo"] for r in records))
L_labels = [f"{v/1e6:.0f}" for v in L_values]

def get_vals(algo, metric, L):
    return [r[metric] for r in records if r["algo"] == algo and r["L_mean"] == L]

print("=" * 100)
print("Exp2_L 实验结果分析: 任务负载 L_mean 对系统性能的影响")
print("=" * 100)
print(f"\nL_mean 取值: {L_labels} Mbit  |  对比算法: {algos}  |  每种配置种子数: {len(set(r['seed'] for r in records))}")

# ---- Table 1 ----
print("\n" + "=" * 100)
print("表1: 各算法在不同 L_mean 下的性能汇总 (均值 +/- 标准差)")
print("=" * 100)

metrics = [("paoi_mean", "PAoI"), ("q_mean", "Queue (bits)"),
           ("e_bs_mean", "E_BS (J)"), ("e_sat_mean", "E_Sat (J)")]

for mkey, mname in metrics:
    print(f"\n--- {mname} ---")
    header = f"{'L_mean':>10}"
    for algo in algos:
        header += f"  {algo:>22}"
    print(header)
    print("-" * len(header))
    for L in L_values:
        row = f"{L/1e6:>6.0f} Mbit"
        for algo in algos:
            vals = get_vals(algo, mkey, L)
            if vals:
                row += f"  {np.mean(vals):>12.4f} +- {np.std(vals):<8.4f}"
            else:
                row += f"  {'N/A':>22}"
        print(row)

# ---- Best algorithm ----
print("\n" + "=" * 100)
print("表2: 各 L_mean 下 PAoI 最优算法")
print("=" * 100)
for L in L_values:
    best_algo, best_val = None, float("inf")
    for algo in algos:
        vals = get_vals(algo, "paoi_mean", L)
        if vals and np.mean(vals) < best_val:
            best_val, best_algo = np.mean(vals), algo
    # Also show LDA vs AC comparison
    lda_m = np.mean(get_vals("LDA", "paoi_mean", L)) if get_vals("LDA", "paoi_mean", L) else None
    ac_m = np.mean(get_vals("AC", "paoi_mean", L)) if get_vals("AC", "paoi_mean", L) else None
    gap = ""
    if lda_m and ac_m:
        pct = (lda_m - ac_m) / ac_m * 100
        gap = f"  (LDA vs AC: {pct:+.1f}%)"
    print(f"  L_mean={L/1e6:.0f} Mbit: PAoI最优={best_algo} ({best_val:.4f}){gap}")

# ---- Trend analysis ----
print("\n" + "=" * 100)
print("表3: L_mean 递增时的指标变化趋势")
print("=" * 100)
for mkey, mname in metrics:
    print(f"\n--- {mname} 逐级变化 ---")
    header = f"{'区间':>12}"
    for algo in algos:
        header += f"  {algo:>22}"
    print(header)
    for i in range(len(L_values) - 1):
        Lf, Lt = L_values[i], L_values[i + 1]
        row = f"{Lf/1e6:.0f}->{Lt/1e6:.0f}"
        for algo in algos:
            vf = np.mean(get_vals(algo, mkey, Lf))
            vt = np.mean(get_vals(algo, mkey, Lt))
            d, pct = vt - vf, (vt - vf) / (abs(vf) + 1e-9) * 100
            row += f"  {d:>+10.4f} ({pct:>+6.1f}%)"
        print(row)

# ---- CoV ----
print("\n" + "=" * 100)
print("表4: 跨种子稳定性 -- PAoI 变异系数 CoV")
print("=" * 100)
header = f"{'L_mean':>10}"
for algo in algos:
    header += f"  {algo:>10}"
print(header)
for L in L_values:
    row = f"{L/1e6:>6.0f} Mbit"
    for algo in algos:
        vals = get_vals(algo, "paoi_mean", L)
        cov = np.std(vals) / (np.mean(vals) + 1e-9) if vals else 0
        row += f"  {cov:>8.4f}"
    print(row)

# ---- Energy compliance ----
print("\n" + "=" * 100)
print("表5: 能量约束合规性 (E_BS_max=180J, E_Sat_max=80J)")
print("=" * 100)
for L in L_values:
    for algo in algos:
        e_bs = np.mean(get_vals(algo, "e_bs_mean", L))
        e_sat = np.mean(get_vals(algo, "e_sat_mean", L))
        bs_ok = "OK" if e_bs <= 180 else "OVER"
        sat_ok = "OK" if e_sat <= 80 else "OVER"
        print(f"  L={L/1e6:.0f}Mbit {algo:>4}: E_BS={e_bs:6.1f}J [{bs_ok}]  E_Sat={e_sat:6.1f}J [{sat_ok}]")

# ---- LDA vs AC ablation (full breakdown) ----
print("\n" + "=" * 100)
print("表6: 消融分析 -- LDA vs AC (PAoI项贡献)")
print("=" * 100)
print(f"{'L_mean':>10}  {'LDA_PAoI':>10}  {'AC_PAoI':>10}  {'Delta':>10}  {'PAoI提升%':>10}  {'LDA_Q':>12}  {'AC_Q':>12}")
for L in L_values:
    lda_p = np.mean(get_vals("LDA", "paoi_mean", L))
    ac_p = np.mean(get_vals("AC", "paoi_mean", L))
    lda_q = np.mean(get_vals("LDA", "q_mean", L))
    ac_q = np.mean(get_vals("AC", "q_mean", L))
    delta = ac_p - lda_p
    imp = delta / ac_p * 100
    print(f"{L/1e6:>6.0f} Mbit  {lda_p:>10.4f}  {ac_p:>10.4f}  {delta:>+10.4f}  {imp:>+9.1f}%  {lda_q:>12.1f}  {ac_q:>12.1f}")

print("\n" + "=" * 100)
print("分析完成")
print("=" * 100)
