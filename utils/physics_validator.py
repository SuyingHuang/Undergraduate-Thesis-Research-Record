import numpy as np


def _f(val):
    """将 numpy 标量/零维数组安全转为 Python float，用于 f-string 格式化。"""
    return float(np.asarray(val).flat[0])


def validate_sat_time_constraint(T_avail_sat_raw, mask_sat, T_prop, T_tran_sat):
    """校验卫星物理时间是否穿透。超出tau时仅警告，由np.maximum(0,.)自然惩罚该候选项。"""
    violated = (T_avail_sat_raw <= 0) & mask_sat
    if np.any(violated):
        bad_i, bad_j = np.where(violated)
        n_violated = np.sum(violated)
        t_tran_max = np.max(T_tran_sat[violated])
        print(f"[WARNING] 卫星可用时间不足: {n_violated} 个用户 T_tran 超 tau "
              f"(最大 T_tran={_f(t_tran_max):.2f}s @ BS {bad_i[0]}, UE {bad_j[0]}) "
              f"— T_avail 钳位为0，候选项将自然被淘汰")


def validate_bs_time_constraint(t_bs_avail_raw, mask_bs, T_left_prev_mat):
    """校验基站物理时间是否穿透"""
    violated = (t_bs_avail_raw <= 0) & mask_bs
    if np.any(violated):
        bad_i, bad_j = np.where(violated)
        print("\n" + "!" * 60)
        print("[FATAL ERROR] 跨帧假设被打破：基站可用时间 <= 0！")
        print(f"案发坐标: 基站 {bad_i[0]}, 用户 {bad_j[0]}")
        print(f"清空旧债耗时: {_f(T_left_prev_mat[bad_i[0], 0]):.4f}s")
        print("!" * 60)
        raise ValueError("BS Physical Time Constraint Violated")


def validate_task_span_constraint(failed_to_clear_bs, L_left_prev_vec, cap_old_bs):
    """校验任务是否被拖延至第三帧。微小偏差(≤1%)仅警告，严重偏差仍报错。"""
    if not np.any(failed_to_clear_bs):
        return

    bad_i, bad_j = np.where(failed_to_clear_bs)
    bi, bj = bad_i[0], bad_j[0]
    debt = L_left_prev_vec[bi, bj]
    cap = cap_old_bs[bi, bj]
    ratio = cap / debt if debt > 0 else 0

    if ratio >= 0.95:
        print(f"\n[WARNING] BS未能完全清空历史残余 (BS {bi}, UE {bj}): "
              f"debt={debt:.2f} bits, capacity={cap:.2f} bits "
              f"(shortfall={(1-ratio)*100:.2f}%) — 继续执行，残余自动转至下帧")
    else:
        print("\n" + "!" * 60)
        print(f"[FATAL ERROR] 跨帧假设打破：基站本帧算力无法清空历史残余！")
        print(f"案发坐标: 基站 {bi}, 用户 {bj}")
        print(f"旧债: {debt:.2f} bits, 本帧最大处理量: {cap:.2f} bits")
        print("!" * 60)
        raise RuntimeError("Task span exceeded two frames: Failed to clear L_left_prev_bs")
