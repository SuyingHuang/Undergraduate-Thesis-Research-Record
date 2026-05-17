"""
趋势分析工具：加载所有 _metrics.json，分析指标随参数变化的规律
用法: python analyze_metrics.py [exp_dir_name]
示例: python analyze_metrics.py 20260508_004940_Exp1_J
"""
import os, sys, json, glob
import numpy as np
from collections import defaultdict

LOG_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'sweep')

def load_all_metrics(exp_dir):
    """加载一个实验组目录下所有 _metrics.json"""
    pattern = os.path.join(LOG_BASE, exp_dir, '*_metrics.json')
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        # 从文件名解析: algo_paramVal_seed
        basename = os.path.basename(f).replace('_metrics.json', '')
        parts = basename.split('_')
        algo = parts[0]
        # 例如 LDA_J4_s42 → algo=LDA, param=J4
        # 或 COB_J12_s1000
        for i, p in enumerate(parts[1:], 1):
            if p.startswith('s') and p[1:].isdigit():
                param_str = '_'.join(parts[1:i])
                seed_str = p
                break
        param_val = data['meta']['param_val']
        seed = data['meta']['seed']
        records.append(data)
    return records

def analyze_exp(exp_dir):
    """分析单个实验组"""
    records = load_all_metrics(exp_dir)
    if not records:
        print(f"未找到数据: {exp_dir}")
        return

    # --- 1. 基础汇总 ---
    param_name = records[0]['meta']['param_name']
    print(f"\n{'='*70}")
    print(f"  实验组: {exp_dir}")
    print(f"  参数: {param_name}")
    print(f"  总记录数: {len(records)}")
    print(f"{'='*70}")

    # 按 (param_val, algo) 聚合
    groups = defaultdict(list)
    for r in records:
        key = (r['meta']['param_val'], r['meta']['algo'])
        groups[key].append(r)

    # 排序
    sorted_params = sorted(set(k[0] for k in groups.keys()))
    algos = sorted(set(k[1] for k in groups.keys()))

    # --- 2. 指标总览表 ---
    print(f"\n  {'参数':>6s}", end="")
    for algo in algos:
        print(f" | {algo:>6s} PAoI | {algo:>6s} E_BS | {algo:>6s} E_LEO | {algo:>6s} Q ", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in algos:
        print(f"─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}", end="")
    print()

    for pv in sorted_params:
        print(f"  {pv:>6}", end="")
        for algo in algos:
            recs = groups.get((pv, algo), [])
            if recs:
                paoi = np.mean([r['summary']['paoi_mean'] for r in recs])
                e_bs = np.mean([r['summary']['e_bs_mean'] for r in recs])
                e_leo = np.mean([r['summary']['e_sat_mean'] for r in recs])
                q = np.mean([r['summary']['q_mean'] for r in recs])
                paoi_std = np.std([r['summary']['paoi_mean'] for r in recs])
                print(f" | {paoi:>6.3f} | {e_bs:>6.1f} | {e_leo:>6.1f} | {q:>6.3f}", end="")
            else:
                print(f" | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>6s}", end="")
        print()

    # --- 3. 趋势分析: 指标增长率 ---
    print(f"\n{'─'*60}")
    print(f"  趋势分析: 相邻参数值的指标变化率")
    print(f"{'─'*60}")
    for algo in algos:
        print(f"\n  [{algo}]")
        for i in range(len(sorted_params) - 1):
            p1, p2 = sorted_params[i], sorted_params[i+1]
            r1 = groups.get((p1, algo), [])
            r2 = groups.get((p2, algo), [])
            if not r1 or not r2:
                continue
            paoi1 = np.mean([r['summary']['paoi_mean'] for r in r1])
            paoi2 = np.mean([r['summary']['paoi_mean'] for r in r2])
            q1 = np.mean([r['summary']['q_mean'] for r in r1])
            q2 = np.mean([r['summary']['q_mean'] for r in r2])
            ebs1 = np.mean([r['summary']['e_bs_mean'] for r in r1])
            ebs2 = np.mean([r['summary']['e_bs_mean'] for r in r2])
            eleo1 = np.mean([r['summary']['e_sat_mean'] for r in r1])
            eleo2 = np.mean([r['summary']['e_sat_mean'] for r in r2])

            dp = p2 - p1
            print(f"    {p1}→{p2}: "
                  f"ΔPAoI/Δ{param_name}={paoi2-paoi1:+.4f} "
                  f"ΔQ={q2-q1:+.4f} "
                  f"ΔE_BS={ebs2-ebs1:+.1f} "
                  f"ΔE_LEO={eleo2-eleo1:+.1f}")

    # --- 4. 收敛分析 (仅 LDA1, LDA2) ---
    print(f"\n{'─'*60}")
    print(f"  收敛分析: delta_t 探索期统计")
    print(f"{'─'*60}")
    for algo in ['LDA', 'AC']:  # 内部文件名
        disp = 'LDA1' if algo == 'LDA' else 'LDA2'
        print(f"\n  [{disp}]")
        for pv in sorted_params:
            recs = groups.get((pv, algo), [])
            if not recs:
                continue
            conv_frames = [r['meta']['conv_frame'] for r in recs]
            used_frames = [r['meta']['n_used_frames'] for r in recs]
            mean_conv = np.mean(conv_frames)
            mean_used = np.mean(used_frames)
            pct_used = mean_used / recs[0]['meta']['sim_frames'] * 100
            print(f"    {param_name}={pv:>6}: 收敛帧={mean_conv:>6.0f}  "
                  f"使用帧={mean_used:>6.0f} ({pct_used:.0f}%)  "
                  f"[8种子: {conv_frames}]")

    # --- 5. 稳定性分析: 跨种子变异系数 ---
    print(f"\n{'─'*60}")
    print(f"  稳定性分析: 跨种子 CV (变异系数 = std/mean, 越低越稳定)")
    print(f"{'─'*60}")
    for algo in algos:
        cvs = []
        for pv in sorted_params:
            recs = groups.get((pv, algo), [])
            if len(recs) < 2:
                continue
            paoi_vals = [r['summary']['paoi_mean'] for r in recs]
            cv = np.std(paoi_vals) / (np.mean(paoi_vals) + 1e-9)
            cvs.append(cv)
        if cvs:
            print(f"  {algo}: 平均 CV_PAoI = {np.mean(cvs):.4f}  (min={np.min(cvs):.4f}, max={np.max(cvs):.4f})")

    # --- 6. 瓶颈推测: 队列分解 ---
    print(f"\n{'─'*60}")
    print(f"  瓶颈分析: Q_bs vs Q_sat 占比")
    print(f"{'─'*60}")
    for algo in algos:
        print(f"\n  [{algo}]")
        for pv in sorted_params:
            recs = groups.get((pv, algo), [])
            if not recs:
                continue
            # 取第一个种子的轨迹做分析
            r = recs[0]
            traj = r['trajectory']
            q_bs_arr = np.array(traj.get('Q_bs', []))
            q_sat_arr = np.array(traj.get('Q_sat', []))
            if len(q_bs_arr) == 0:
                continue
            # 只用收敛后的帧
            si = r['meta']['start_idx'] // 10  # 采样步长为10
            q_bs_mean = np.mean(q_bs_arr[si:])
            q_sat_mean = np.mean(q_sat_arr[si:])
            total = q_bs_mean + q_sat_mean + 1e-9
            print(f"    {param_name}={pv:>6}: BS={q_bs_mean:.3f} ({q_bs_mean/total*100:.0f}%)  "
                  f"SAT={q_sat_mean:.3f} ({q_sat_mean/total*100:.0f}%)")

    return groups, sorted_params, algos


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_dirs = [sys.argv[1]]
    else:
        # 自动找所有实验组
        exp_dirs = sorted([d for d in os.listdir(LOG_BASE)
                          if os.path.isdir(os.path.join(LOG_BASE, d))])

    for ed in exp_dirs:
        try:
            analyze_exp(ed)
        except Exception as e:
            print(f"\n[ERROR] {ed}: {e}")
            import traceback; traceback.print_exc()
