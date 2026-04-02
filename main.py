# main.py

import numpy as np
import random
import torch
import sys
import os

# 设置系统路径，确保模块可导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from core.env import SAGINEnvironment
from core.agents.lda_agent import LDAAgent
from utils.plotter import plot_results


def set_seed(seed=42):
    """设置全局随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_simulation(cfg, agent_class, algorithm_name="Algorithm"):
    """
    通用实验运行器
    """
    print(f"\n==================================================")
    print(f"   启动仿真实验: {algorithm_name} ")
    print(f"==================================================")

    # 1. 实例化“裁判员” (物理环境) 和 “运动员” (优化算法)
    env = SAGINEnvironment(cfg)
    agent = agent_class(cfg)

    print(">>> 时隙仿真开始执行...")

    for t in range(cfg.sim_frames):
        # A. 环境演进：生成物理考题（信道衰落和业务到达）
        R_bs, R_sat, T_prop = env.generate_channel_states()
        noise = np.random.normal(0, cfg.L_std, (cfg.I, cfg.J))
        L_t = np.maximum(0, cfg.L_mean + noise)

        # B. 算法决策：智能体根据考题和当前环境排队状态，计算策略
        action = agent.select_action(env, L_t, R_bs, R_sat, T_prop)

        # C. 环境步进：裁判员根据算法策略，推进物理时间线，更新队列
        env.step(action, L_t)

        # D. 模型训练：触发强化学习反向传播（如果存在训练方法）
        if hasattr(agent, 'train'):
            agent.train(t)

        # E. 进度监控日志打印
        if t % 50 == 0:
            info = action.get('debug', {})
            q_mb = np.mean(env.Q_total) / 1e6
            max_e_virt = np.max(env.E_BS)

            log_str = f"[Fr {t:04d}] Q: {q_mb:6.1f}Mb | Max E_virt: {max_e_virt:6.1f}"

            if info:
                n_loc, n_bs, n_sat = info['dist']
                arr, srv = info['flow']
                trend_symbol = "🟢" if info['q_trend'] > 0 else "🔴"
                log_str += f" {trend_symbol} | In/Out: {arr:4.1f}/{srv:4.1f} | Dec(L/B/S): {n_loc}/{n_bs}/{n_sat}"

            print(log_str)

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