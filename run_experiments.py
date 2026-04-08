import os
import random
import numpy as np
import pickle
import torch

from config import SystemConfig
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import ACAgent, COBAgent, MTDAgent
from main import run_simulation  # 复用你 main.py 里的核心仿真循环
from utils.plotter import plot_results_comparison  # 我们稍后在 plotter 里添加这个函数


def set_seed(seed=42):
    """
    固定所有的随机种子，保证每次环境生成的任务量、信道状态完全一致，
    从而确保不同算法面临的“考卷”是一样的，保证绝对公平。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(history, algo_name, save_dir="results"):
    """保存单个算法的仿真结果"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f"simulation_results_{algo_name}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"[{algo_name}] 的仿真数据已保存至: {file_path}")


def main():
    cfg = SystemConfig()

    # 你可以在这里临时覆盖 config 中的仿真帧数，用于快速测试
    # cfg.sim_frames = 500

    algorithms = [
        ("LDA", LDAAgent),
        ("AC", ACAgent),
        ("COB", COBAgent),
        ("MTD", MTDAgent)
    ]

    all_histories = {}
    base_seed = 42  # 定义一个全局基准种子

    print("=" * 50)
    print("开始执行多算法公平对比实验...")
    print("=" * 50)

    for algo_name, AgentClass in algorithms:
        print(f"\n>>> 正在运行算法: {algo_name}")

        # 【关键】在初始化每个环境前，重置为相同的种子！
        set_seed(base_seed)

        # 运行仿真 (假设 run_simulation 返回 environment 和 agent)
        env, agent = run_simulation(cfg, AgentClass, algorithm_name=algo_name)

        # 提取你想记录的数据 (这里假设 env.history 记录了每帧的各项指标)
        # 如果 LDA 还有单独的 loss，也可以合并进去
        history = env.history
        if hasattr(agent, 'loss_history') and algo_name == "LDA":
            history['Loss'] = agent.loss_history

        all_histories[algo_name] = history

        # 自动保存该算法的独立结果
        save_results(history, algo_name)

    print("\n" + "=" * 50)
    print("所有算法运行完毕，正在生成综合对比图...")
    print("=" * 50)

    # 绘制对比图并保存
    plot_results_comparison(all_histories, save_path="results/final_comparison_plot.png")
    print("实验全部完成！所有结果均已存入 results/ 文件夹。")


if __name__ == "__main__":
    main()