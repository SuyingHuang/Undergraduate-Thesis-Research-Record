import numpy as np
import config


def check_local_feasibility(task_sizes, local_freqs):
    """
    根据 Eq.(9) 判断任务是否必须卸载。
    如果本地计算时间 > 帧长 tau，则必须卸载 (l=0)；
    否则优先本地计算 (l=1)。

    Args:
        task_sizes (np.array): 任务大小 L_t,ij (bits), shape (J,)
        local_freqs (np.array): 本地计算频率 f_l (Hz), shape (J,)

    Returns:
        l_decisions (np.array): 0 表示必须卸载，1 表示本地执行
    """
    # 计算本地处理时间: T = (phi * L) / f_l
    cfg = config.SystemConfig()
    local_times = (cfg.phi* task_sizes) / local_freqs

    # 如果时间超过 tau，则 l=0 (False -> 0), 否则 l=1 (True -> 1)
    # 注意：论文 Eq.(9) 说 T > tau 则 l=0 (offload)
    # 这里的逻辑是：can_local = T <= tau
    can_local = local_times <= cfg.tau
    return can_local.astype(int)


def generate_candidates(dnn_output, delta_t, l_decisions):
    """
    实现 TCOPQ 算法 (Algorithm 1, Line 7-8)

    Args:
        dnn_output (np.array): DNN 输出的连续值 \tilde{b}, shape (J,)
        delta_t (float): 当前的量化阈值 delta
        l_decisions (np.array): 确定的本地决策向量 l

    Returns:
        candidates (list): 包含多个元组 (l_vec, b_vec) 的列表
    """
    J = len(dnn_output)
    candidates = []

    # --- 1. 生成基准决策 (Standard Rounding) ---
    # 大于 0.5 设为 1 (BS)，否则设为 0 (LEOS)
    b_base = (dnn_output >= 0.5).astype(int)#基准卸载决策

    # 只有当 l=0 (必须卸载) 时，b 的值才有意义
    # 但为了格式统一，我们可以先生成完整的 b 向量
    candidates.append((l_decisions, b_base))

    # --- 2. TCOPQ 生成额外候选 ---
    # 计算每个决策位的不确定度 (距离 0.5 的距离)
    uncertainty = np.abs(dnn_output - 0.5)

    # 排序：距离越小(越接近0.5)，越不确定，排在前面
    sorted_indices = np.argsort(uncertainty)

    # 筛选出满足阈值 delta_t 的不稳定位
    # Eq.(42): |b - 0.5| <= delta_t
    valid_indices = [idx for idx in sorted_indices if uncertainty[idx] <= delta_t]

    # 针对这些不稳定位，生成翻转后的候选决策
    # 策略：按照不确定度顺序，依次翻转该位 (BS <-> LEOS)
    for idx in valid_indices:
        # 如果该任务是本地执行 (l=1)，翻转 b 没有意义，跳过以节省计算
        if l_decisions[idx] == 1:
            continue

        b_new = b_base.copy()
        b_new[idx] = 1 - b_base[idx]  # 翻转状态
        candidates.append((l_decisions, b_new))

    return candidates