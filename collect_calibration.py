"""
标定脚本：多种子 × 前 N 帧（DNN 尚未学习），收集 G1 三项的原始量级，
为 Q_ref / PAoI_ref / E_ref 提供参考值。

思路：参考尺度应反映「无知策略」下各项的天然物理量级，
     而非收敛后的稳态值。这样可以保证：
     - 探索期：三项归一化后量级可比 → 均衡学习信号
     - 收敛后：Q/E 被压到接近 0，PAoI 自然主导优化

用法：python collect_calibration.py [--frames 200] [--n_seeds 5]
"""

import numpy as np
import random
import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from core.env import SAGINEnvironment
from core.agents.lda_agent import LDAAgent


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_raw_terms(env, details, cfg):
    term_q_bs = np.sum((env.Q_bs / 1e5) * ((details['l_left_bs'] - details['l_proc_old_bs']) / 1e4))
    term_q_sat = np.sum((env.Q_sat / 1e5) * ((details['l_left_sat'] - env.current_q_sat_reduction_mat) / 1e4))
    term_q = term_q_bs + term_q_sat
    term_p = cfg.K_p * np.sum(details['paoi'])
    term_e_bs = np.sum(env.E_BS * (details['e_bs_total'] - cfg.E_max_BS))
    return float(term_q), float(term_p), float(term_e_bs)


def run_single_seed(frames, seed, verbose=True):
    """单种子运行，返回 (term_q_arr, term_p_arr, term_e_arr)"""
    set_seed(seed)
    cfg = SystemConfig()
    cfg.sim_frames = frames

    env = SAGINEnvironment(cfg)
    agent = LDAAgent(cfg)

    raw_q, raw_p, raw_e = [], [], []

    for t in range(frames):
        R_bs, R_sat, T_prop = env.generate_channel_states()
        noise = np.random.normal(0, cfg.L_std, (cfg.I, cfg.J))
        L_t = np.maximum(0, cfg.L_mean + noise)

        action = agent.select_action(env, L_t, R_bs, R_sat, T_prop, t=t)
        env.step(action, L_t)

        if hasattr(agent, 'train'):
            agent.train(t)

        if 'details' in action:
            tq, tp, te = compute_raw_terms(env, action['details'], cfg)
            raw_q.append(tq)
            raw_p.append(tp)
            raw_e.append(te)

    tq_arr = np.array(raw_q)
    tp_arr = np.array(raw_p)
    te_arr = np.array(raw_e)

    if verbose:
        print(f"  seed={seed:4d}  |term_q| median={np.median(np.abs(tq_arr)):.2e}  "
              f"|term_p| median={np.median(np.abs(tp_arr)):.2e}  "
              f"|term_e| median={np.median(np.abs(te_arr)):.2e}")

    return tq_arr, tp_arr, te_arr


def run_calibration(frames=200, seeds=None):
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    cfg = SystemConfig()

    print(f"{'='*60}")
    print(f"  G1 三项量级标定")
    print(f"  策略：多种子 × 前 {frames} 帧（DNN 未学习，纯探索期）")
    print(f"  种子数：{len(seeds)}  |  每种子帧数：{frames}")
    print(f"  总样本：{len(seeds) * frames} 帧")
    print(f"  E_max_BS = {cfg.E_max_BS}")
    print(f"{'='*60}\n")

    all_q, all_p, all_e = [], [], []

    for seed in seeds:
        tq, tp, te = run_single_seed(frames, seed, verbose=True)
        all_q.append(tq)
        all_p.append(tp)
        all_e.append(te)

    # 汇总所有种子
    stack_q = np.concatenate(all_q)
    stack_p = np.concatenate(all_p)
    stack_e = np.concatenate(all_e)

    print(f"\n{'='*60}")
    print(f"  各种子统计")
    print(f"{'='*60}")
    print(f"  {'seed':>6}  {'|term_q| median':>16}  {'|term_p| median':>16}  {'|term_e| median':>16}")
    print(f"  {'-'*62}")
    for i, seed in enumerate(seeds):
        print(f"  {seed:>6}  {np.median(np.abs(all_q[i])):>16.4e}  "
              f"{np.median(np.abs(all_p[i])):>16.4e}  "
              f"{np.median(np.abs(all_e[i])):>16.4e}")

    # 跨种子汇总统计
    print(f"\n{'='*60}")
    print(f"  汇总标定结果  (总帧数: {len(stack_q)})")
    print(f"{'='*60}")

    for name, arr in [('term_q (队列漂移)', stack_q),
                       ('term_p (PAoI惩罚)', stack_p),
                       ('term_e (能量漂移)', stack_e)]:
        abs_arr = np.abs(arr)
        print(f"\n  [{name}]")
        print(f"    原始值  Mean: {np.mean(arr):>14.4e}   Median: {np.median(arr):>14.4e}")
        print(f"    原始值  Std:  {np.std(arr):>14.4e}   Min: {np.min(arr):>14.4e}   Max: {np.max(arr):>14.4e}")
        print(f"    绝对值  Mean: {np.mean(abs_arr):>14.4e}   Median: {np.median(abs_arr):>14.4e}")
        print(f"    百分位  P25: {np.percentile(arr, 25):>14.4e}   P75: {np.percentile(arr, 75):>14.4e}")

    # 跨种子中位数的中位数（更稳健）
    per_seed_med_q = [np.median(np.abs(a)) for a in all_q]
    per_seed_med_p = [np.median(np.abs(a)) for a in all_p]
    per_seed_med_e = [np.median(np.abs(a)) for a in all_e]

    abs_med_q = np.median(per_seed_med_q)
    abs_med_p = np.median(per_seed_med_p)
    abs_med_e = np.median(per_seed_med_e)

    print(f"\n{'='*60}")
    print(f"  推荐参考尺度 (跨种子中位数之中位数)")
    print(f"{'='*60}")
    print(f"  各种子 |term_q| median: {[f'{v:.4e}' for v in per_seed_med_q]}")
    print(f"  各种子 |term_p| median: {[f'{v:.4e}' for v in per_seed_med_p]}")
    print(f"  各种子 |term_e| median: {[f'{v:.4e}' for v in per_seed_med_e]}")
    print(f"")
    print(f"  当前 config.py:  Q_ref={cfg.Q_ref:.2g}  PAoI_ref={cfg.PAoI_ref:.2g}  E_ref={cfg.E_ref:.2g}")
    print(f"  推荐更新为:      Q_ref={abs_med_q:.6g}  PAoI_ref={abs_med_p:.6g}  E_ref={abs_med_e:.6g}")

    # 归一化验证 (全量)
    eps = 1e-12
    norm_q = stack_q / (abs_med_q + eps)
    norm_p = stack_p / (abs_med_p + eps)
    norm_e = stack_e / (abs_med_e + eps)

    print(f"\n  [归一化后统计 (均值 +/- 标准差)]")
    print(f"  norm_term_q:  {np.mean(norm_q):+.4f} +/- {np.std(norm_q):.4f}")
    print(f"  norm_term_p:  {np.mean(norm_p):+.4f} +/- {np.std(norm_p):.4f}")
    print(f"  norm_term_e:  {np.mean(norm_e):+.4f} +/- {np.std(norm_e):.4f}")

    total_abs = np.abs(norm_q) + np.abs(norm_p) + np.abs(norm_e) + eps
    share_q = np.mean(np.abs(norm_q) / total_abs)
    share_p = np.mean(np.abs(norm_p) / total_abs)
    share_e = np.mean(np.abs(norm_e) / total_abs)
    print(f"\n  [三项平均占比]")
    print(f"  Q-term:  {share_q*100:.1f}%")
    print(f"  P-term:  {share_p*100:.1f}%")
    print(f"  E-term:  {share_e*100:.1f}%")

    return stack_q, stack_p, stack_e, (abs_med_q, abs_med_p, abs_med_e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=200,
                        help='每个种子的帧数 (默认 200，此时 DNN 尚未开始训练)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='种子数量 (默认 5)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='手动指定种子列表，逗号分隔，如 "42,123,456"')
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = [42 + i * 100 for i in range(args.n_seeds)]

    run_calibration(frames=args.frames, seeds=seeds)
