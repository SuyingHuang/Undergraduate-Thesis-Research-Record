import numpy as np


def check_local_feasibility(task_sizes, local_freqs, cfg):
    """
    判断每个任务是否能在单帧内完成本地计算。

    Returns:
        l_decisions: 1 表示本地执行，0 表示必须卸载。
    """
    local_times = (cfg.phi * task_sizes) / local_freqs
    can_local = local_times <= cfg.tau
    return can_local.astype(int)


def generate_candidates(dnn_output, delta_t, l_decisions):
    """
    根据 DNN 输出的卸载概率生成 TCOPQ 候选决策。

    第一个候选是标准四舍五入结果。额外候选同时包含两类：
    1. 逐位翻转候选：每个候选只翻转一个不确定位；
    2. 保序累积翻转候选：第 k 个候选翻转前 k 个最不确定的可卸载位。

    Args:
        dnn_output: 单个 BS 的 DNN 连续输出，形状为 (J,)。
        delta_t: 以 0.5 为中心的不确定性窗口。
        l_decisions: 本地/卸载决策，1 表示本地执行，0 表示必须卸载。

    Returns:
        由 (l_vec, b_vec) 元组组成的候选列表。
    """
    candidates = []
    seen = set()

    def add_candidate(b_vec):
        key = tuple(np.asarray(b_vec, dtype=int).tolist())
        if key not in seen:
            seen.add(key)
            candidates.append((l_decisions, np.asarray(b_vec, dtype=int).copy()))

    # 标准量化：b=1 表示卸载到 BS，b=0 表示卸载到 LEOS。
    b_base = (dnn_output >= 0.5).astype(int)
    add_candidate(b_base)

    uncertainty = np.abs(dnn_output - 0.5)
    sorted_indices = np.argsort(uncertainty)

    valid_indices = [
        idx for idx in sorted_indices
        if uncertainty[idx] <= delta_t and l_decisions[idx] == 0
    ]

    # 原始逐位候选：每次只翻转一个不确定位。
    for idx in valid_indices:
        b_single = b_base.copy()
        b_single[idx] = 1 - b_single[idx]
        add_candidate(b_single)

    # 保序累积候选：按不确定性顺序翻转前 k 个不确定位。
    b_running = b_base.copy()
    for idx in valid_indices:
        b_running[idx] = 1 - b_running[idx]
        add_candidate(b_running)

    return candidates
