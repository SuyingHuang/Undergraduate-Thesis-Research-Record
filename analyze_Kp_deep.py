"""
深度分析：为什么 K_p 越大 PAoI 反而越差？
从轨迹层面诊断 K_p=0.1 vs K_p=1 的行为差异
"""
import json, glob, numpy as np

EXP_DIR = "E:/PY_Project/LDA/logs/sweep/20260509_190140_Exp4_K"

def load_trajectories(algo, k_val, seed=42):
    """加载指定配置的轨迹数据"""
    pattern = f"{EXP_DIR}/{algo}_K_p{k_val}_s{seed}_metrics.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  NOT FOUND: {pattern}")
        return None
    with open(files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_pair(algo, seed=42):
    """对比 K_p=0.1 vs K_p=1 的完整轨迹"""
    d_lo = load_trajectories(algo, 0.1, seed)
    d_hi = load_trajectories(algo, 1, seed)
    if d_lo is None or d_hi is None:
        return

    traj_lo = d_lo['trajectory']
    traj_hi = d_hi['trajectory']
    meta_lo = d_lo['meta']
    meta_hi = d_hi['meta']

    si_lo = meta_lo['start_idx'] // 10  # 收敛后采样索引
    si_hi = meta_hi['start_idx'] // 10

    # 收敛后的帧
    paoi_lo = np.array(traj_lo['Cost'][si_lo:])
    paoi_hi = np.array(traj_hi['Cost'][si_hi:])
    q_lo = np.array(traj_lo['Q_total'][si_lo:])
    q_hi = np.array(traj_hi['Q_total'][si_hi:])
    e_lo = np.array(traj_lo['E_virt_bs'][si_lo:])
    e_hi = np.array(traj_hi['E_virt_bs'][si_hi:])

    print(f"\n{'='*65}")
    print(f"  [{algo}] K_p=0.1 vs K_p=1, seed={seed}")
    print(f"{'='*65}")
    print(f"  收敛后帧数: K=0.1: {len(paoi_lo):>5d}  |  K=1: {len(paoi_hi):>5d}")

    # 1. 均值对比
    print(f"\n  ── 均值对比 ──")
    print(f"  {'指标':<12s} {'K_p=0.1':>10s} {'K_p=1':>10s} {'Δ':>10s}")
    print(f"  PAoI:       {np.mean(paoi_lo):10.4f} {np.mean(paoi_hi):10.4f} {np.mean(paoi_hi)-np.mean(paoi_lo):+10.4f}")
    print(f"  Q_total:    {np.mean(q_lo)/1e6:10.4f}M {np.mean(q_hi)/1e6:10.4f}M {(np.mean(q_hi)-np.mean(q_lo))/1e6:+10.4f}M")
    print(f"  E_BS:       {np.mean(e_lo):10.1f} {np.mean(e_hi):10.1f} {np.mean(e_hi)-np.mean(e_lo):+10.1f}")

    # 2. 稳定性对比（变异系数）
    cv_lo = np.std(paoi_lo) / (np.mean(paoi_lo) + 1e-9)
    cv_hi = np.std(paoi_hi) / (np.mean(paoi_hi) + 1e-9)
    print(f"  PAoI CV:    {cv_lo:10.4f} {cv_hi:10.4f}")

    # 3. PAoI 分布
    pcts = [10, 25, 50, 75, 90]
    print(f"\n  ── PAoI 分位数 ──")
    print(f"  {'分位':<8s}", end="")
    for p in pcts:
        print(f" {'P'+str(p):>8s}", end="")
    print()
    print(f"  {'K=0.1':<8s}", end="")
    for p in pcts:
        print(f" {np.percentile(paoi_lo, p):8.2f}", end="")
    print()
    print(f"  {'K=1':<8s}", end="")
    for p in pcts:
        print(f" {np.percentile(paoi_hi, p):8.2f}", end="")
    print()

    # 4. PAoI 时间序列趋势（分 4 段）
    print(f"\n  ── PAoI 时间演化 (4等分段均值) ──")
    n = min(len(paoi_lo), len(paoi_hi))
    seg_len = n // 4
    for s in range(4):
        lo_seg = np.mean(paoi_lo[s*seg_len:(s+1)*seg_len])
        hi_seg = np.mean(paoi_hi[s*seg_len:(s+1)*seg_len])
        print(f"    Seg {s+1}: K=0.1={lo_seg:.4f}  |  K=1={hi_seg:.4f}  |  Δ={hi_seg-lo_seg:+.4f}")

    # 5. Q 与 PAoI 的滞后相关性
    print(f"\n  ── Q 与 PAoI 的协同分析 ──")
    # 检查高 PAoI 帧是否与高 Q 帧同步
    paoi_hi_z = (paoi_hi - np.mean(paoi_hi)) / np.std(paoi_hi)
    q_hi_z = (q_hi - np.mean(q_hi)) / np.std(q_hi)
    corr = np.corrcoef(paoi_hi_z, q_hi_z)[0, 1]
    print(f"    K=1: corr(PAoI, Q) = {corr:.4f}")

    paoi_lo_z = (paoi_lo - np.mean(paoi_lo)) / np.std(paoi_lo)
    q_lo_z = (q_lo - np.mean(q_lo)) / np.std(q_lo)
    corr_lo = np.corrcoef(paoi_lo_z, q_lo_z)[0, 1]
    print(f"    K=0.1: corr(PAoI, Q) = {corr_lo:.4f}")

    # 6. 队列爆炸频率（Q 超过阈值的帧占比）
    for thresh_mb in [2, 5, 10]:
        lo_pct = np.mean(q_lo > thresh_mb * 1e6) * 100
        hi_pct = np.mean(q_hi > thresh_mb * 1e6) * 100
        print(f"    Q > {thresh_mb}Mb: K=0.1={lo_pct:.1f}%  K=1={hi_pct:.1f}%")

    # 7. E_BS 虚拟能量队列分析
    print(f"\n  ── E_BS 能量队列分析 ──")
    print(f"    K=0.1: mean={np.mean(e_lo):.1f}  max={np.max(e_lo):.1f}  P99={np.percentile(e_lo, 99):.1f}")
    print(f"    K=1:   mean={np.mean(e_hi):.1f}  max={np.max(e_hi):.1f}  P99={np.percentile(e_hi, 99):.1f}")

    # 8. Q_bs vs Q_sat 分解
    if 'Q_bs' in traj_lo:
        qbs_lo = np.array(traj_lo['Q_bs'][si_lo:])
        qsat_lo = np.array(traj_lo['Q_sat'][si_lo:])
        qbs_hi = np.array(traj_hi['Q_bs'][si_hi:])
        qsat_hi = np.array(traj_hi['Q_sat'][si_hi:])
        print(f"\n  ── 队列分解 ──")
        print(f"    K=0.1: Q_bs={np.mean(qbs_lo)/1e6:.3f}M  Q_sat={np.mean(qsat_lo)/1e6:.3f}M")
        print(f"    K=1:   Q_bs={np.mean(qbs_hi)/1e6:.3f}M  Q_sat={np.mean(qsat_hi)/1e6:.3f}M")

    return {
        'algo': algo,
        'paoi_lo': float(np.mean(paoi_lo)),
        'paoi_hi': float(np.mean(paoi_hi)),
        'q_lo': float(np.mean(q_lo)),
        'q_hi': float(np.mean(q_hi)),
        'e_lo': float(np.mean(e_lo)),
        'e_hi': float(np.mean(e_hi)),
        'cv_lo': float(cv_lo),
        'cv_hi': float(cv_hi),
        'corr_lo': float(corr_lo),
        'corr_hi': float(corr),
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  K_p 深度诊断：为什么 PAoI 权重越大效果越差？")
    print("=" * 65)

    # 对 LDA 分析 3 个种子
    for algo in ['LDA', 'AC', 'COB']:
        results = []
        for seed in [42, 123, 456]:
            r = analyze_pair(algo, seed)
            if r:
                results.append(r)

        if results:
            print(f"\n  ── [{algo}] 多种子汇总 ──")
            for key, label in [('paoi_lo', 'PAoI(K=0.1)'), ('paoi_hi', 'PAoI(K=1)'),
                                ('q_lo', 'Q(K=0.1)M'), ('q_hi', 'Q(K=1)M'),
                                ('cv_lo', 'CV(K=0.1)'), ('cv_hi', 'CV(K=1)'),
                                ('corr_lo', 'corr(K=0.1)'), ('corr_hi', 'corr(K=1)')]:
                vals = [r[key] for r in results]
                print(f"    {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # 汇总结论
    print(f"\n{'='*65}")
    print(f"  诊断结论")
    print(f"{'='*65}")
    print(f"""
  假设：K_p 增大 → Lyapunov 优化过分追求"即时 PAoI 最小化"
       → 贪婪地给当前帧分配过高 BS 频率 → E_BS 虚拟队列积累
       → 后续帧受能量约束被迫降低频率 → 队列堆积爆炸
       → 长期 PAoI 反而恶化

  预期数据特征：
    1. K_p=1 的 Q_total 应显著高于 K_p=0.1
    2. K_p=1 的 E_BS 应显著高于 K_p=0.1（能量透支）
    3. K_p=1 的 PAoI 方差应更大（队列爆炸—恢复的振荡）
    4. K_p=1 的 Q-PAoI 相关性应更高（队列是 PAoI 的驱动因素）
    5. COB/MTD 不应有此效应（固定策略），但其数据中是否真实存在？
""")
