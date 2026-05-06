# main.py

import numpy as np
import random
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from core.env import SAGINEnvironment
from core.agents.lda_agent import LDAAgent
from utils.plotter import plot_results

E_MAX_BS = 160.0
E_ANOMALY_THRESHOLD = E_MAX_BS * 10    # 超过此值触发详细快照


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _dump_energy_snapshot(env, action, L_t, R_bs, t, first_time=False):
    """
    能量异常时 dump 当前帧的完整频率分配和能耗明细。
    first_time=True 时打印全量，否则打印摘要。
    """
    I, J = env.cfg.I, env.cfg.J
    details = action.get('details', {})
    f_bs = action.get('f_bs', np.zeros((I, J)))
    f_sat = action.get('f_sat', np.zeros((I, J)))
    l_mat = action.get('l', np.zeros((I, J)))
    b_mat = action.get('b', np.zeros((I, J)))
    e_bs_total = details.get('e_bs_total', np.zeros(I))
    l_proc_bs = details.get('l_proc_bs', np.zeros((I, J)))
    l_left_bs = details.get('l_left_bs', np.zeros((I, J)))

    header = "[DBG] ENERGY SNAPSHOT (FIRST)" if first_time else "[DBG] ENERGY SNAPSHOT"
    print(f"\n{'─'*65}")
    print(f"  {header} @ Fr {t}")
    print(f"{'─'*65}")

    # 逐 BS 分析
    for i in range(I):
        e_q = float(env.E_BS[i])
        e_cons = float(e_bs_total[i])
        e_budget = env.cfg.E_max_BS

        # 该 BS 有多少用户卸载到 BS / 卫星 / 本地
        n_bs = int(np.sum((l_mat[i] == 0) & (b_mat[i] == 1)))
        n_sat = int(np.sum((l_mat[i] == 0) & (b_mat[i] == 0)))
        n_loc = int(np.sum(l_mat[i] == 1))

        l_to_bs = float(np.sum(np.where((l_mat[i] == 0) & (b_mat[i] == 1), L_t[i], 0.0)) / 1e6)
        f_bs_mean = float(np.mean(f_bs[i][f_bs[i] > 1e-9])) if np.any(f_bs[i] > 1e-9) else 0.0
        f_bs_max = float(np.max(f_bs[i]))
        l_proc = float(np.sum(l_proc_bs[i]) / 1e6)
        l_left = float(np.sum(l_left_bs[i]) / 1e6)

        marker = " [!!]" if e_q > E_ANOMALY_THRESHOLD else ""
        print(f"  BS{i} | E_q={e_q:8.1f}{marker} | 能耗={e_cons:.1f}J (预算{e_budget:.0f}J)")
        print(f"       | 卸载: {n_bs}BS/{n_sat}SAT/{n_loc}LOC"
              f" | 数据量: {l_to_bs:.1f}Mb→BS")
        print(f"       | f_BS 均值={f_bs_mean/1e6:.1f}MHz  max={f_bs_max/1e6:.1f}MHz"
              f" | 本帧处理={l_proc:.1f}Mb  剩余={l_left:.1f}Mb")

        if first_time:
            # 首次触发：列出该 BS 所有 BS 卸载用户的具体频率
            bs_users = np.where((l_mat[i] == 0) & (b_mat[i] == 1))[0]
            if len(bs_users) > 0:
                f_vals = " ".join(f"{float(f_bs[i,j])/1e6:.1f}" for j in bs_users)
                r_vals = " ".join(f"{float(R_bs[i,j])/1e6:.1f}" for j in bs_users)
                print(f"       | f(MHz) per-user: [{f_vals}]")
                print(f"       | R(Mbps)per-user: [{r_vals}]")

    # 卫星侧
    n_sat_total = int(np.sum((l_mat == 0) & (b_mat == 0)))
    if n_sat_total > 0:
        f_sat_active = f_sat[(l_mat == 0) & (b_mat == 0) & (f_sat > 1e-9)]
        f_s_mean = float(np.mean(f_sat_active)) if len(f_sat_active) > 0 else 0.0
        l_to_sat = float(np.sum(np.where((l_mat == 0) & (b_mat == 0), L_t, 0.0)) / 1e6)
        print(f"  SAT  | 用户数={n_sat_total} | 卸载量={l_to_sat:.1f}Mb | f_SAT均值={f_s_mean/1e6:.1f}MHz")
    print(f"{'─'*65}\n")


def run_simulation(cfg, agent_class, algorithm_name="Algorithm", agent_kwargs=None):
    """
    通用实验运行器
    :param agent_kwargs: dict, 可选的agent属性字典（如 {'lambda_p': 2.0}）
    """
    print(f"\n==================================================")
    print(f"   启动仿真实验: {algorithm_name}")
    print(f"==================================================")

    env = SAGINEnvironment(cfg)
    agent = agent_class(cfg)

    if agent_kwargs:
        for key, val in agent_kwargs.items():
            setattr(agent, key, val)

    print(">>> 时隙仿真开始执行...")

    # 能量异常快照状态
    _anomaly_first_logged = False
    _anomaly_last_logged_frame = -9999

    for t in range(cfg.sim_frames):
        R_bs, R_sat, T_prop = env.generate_channel_states()
        noise = np.random.normal(0, cfg.L_std, (cfg.I, cfg.J))
        L_t = np.maximum(0, cfg.L_mean + noise)

        action = agent.select_action(env, L_t, R_bs, R_sat, T_prop, t=t)

        env.step(action, L_t)

        if hasattr(agent, 'train'):
            agent.train(t)

        # 进度日志 + 能量异常快照
        if t % 50 == 0:
            info = action.get('debug', {})
            q_mb = float(np.mean(env.Q_total) / 1e6)
            max_e_virt = float(np.max(env.E_BS))

            energy_flag = f" [ENERGY_HIGH: {max_e_virt:.0f}]" if max_e_virt > E_ANOMALY_THRESHOLD else ""
            log_str = f"[Fr {t:04d}] Q: {q_mb:6.1f}Mb | Max E_virt: {max_e_virt:6.1f}{energy_flag}"

            if info:
                n_loc, n_bs, n_sat = info['dist']
                arr, srv = float(info['flow'][0]), float(info['flow'][1])
                trend_symbol = "[+]" if info['q_trend'] > 0 else "[-]"
                log_str += f" {trend_symbol} | In/Out: {arr:4.1f}/{srv:4.1f} | Dec(L/B/S): {n_loc}/{n_bs}/{n_sat}"

            print(log_str)

            # 能量异常 → dump 快照
            if max_e_virt > E_ANOMALY_THRESHOLD:
                if not _anomaly_first_logged:
                    _anomaly_first_logged = True
                    _dump_energy_snapshot(env, action, L_t, R_bs, t, first_time=True)
                elif t - _anomaly_last_logged_frame >= 500:
                    _dump_energy_snapshot(env, action, L_t, R_bs, t, first_time=False)
                _anomaly_last_logged_frame = t

    print("\n>>> 仿真结束。")
    return env, agent


if __name__ == "__main__":
    set_seed(42)
    cfg = SystemConfig()

    # [开发调试] 如果跑的太慢，可以把测试帧数暂时改小
    # cfg.sim_frames = 500

    # 1. 运行主算法 LDA
    lda_env, lda_agent = run_simulation(cfg, LDAAgent, algorithm_name="LDA (DRL-based)")

    # 将模型训练过程中的 Loss 塞入环境的 history 中，方便画图脚本统一处理
    lda_env.history['Loss'] = lda_agent.loss_history

    # 2. 生成分析图表
    print(">>> 正在生成分析图表...")
    plot_results(lda_env.history, cfg, save_path='simulation_results_lda.png')