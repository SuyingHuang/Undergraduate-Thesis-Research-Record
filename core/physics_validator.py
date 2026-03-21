import numpy as np

def validate_sat_time_constraint(T_avail_sat_raw, mask_sat, T_prop, T_tran_sat):
    """校验卫星物理时间是否穿透"""
    violated = (T_avail_sat_raw <= 0) & mask_sat
    if np.any(violated):
        bad_i, bad_j = np.where(violated)
        print("\n" + "!"*60)
        print(f"[FATAL ERROR] 跨帧假设被打破：卫星可用时间 <= 0！")
        print(f"案发坐标: 基站 {bad_i[0]}, 用户 {bad_j[0]}")
        print(f"传播延迟 T_prop: {T_prop:.4f}s, 传输耗时 T_tran: {T_tran_sat[bad_i[0], bad_j[0]]:.4f}s")
        print("!"*60)
        raise ValueError("Satellite Physical Time Constraint Violated")

def validate_bs_time_constraint(t_bs_avail_raw, mask_bs, T_left_prev_mat):
    """校验基站物理时间是否穿透"""
    violated = (t_bs_avail_raw <= 0) & mask_bs
    if np.any(violated):
        bad_i, bad_j = np.where(violated)
        print("\n" + "!"*60)
        print(f"[FATAL ERROR] 跨帧假设被打破：基站可用时间 <= 0！")
        print(f"案发坐标: 基站 {bad_i[0]}, 用户 {bad_j[0]}")
        print(f"清空旧债耗时: {T_left_prev_mat[bad_i[0], 0]:.4f}s")
        print("!"*60)
        raise ValueError("BS Physical Time Constraint Violated")

def validate_task_span_constraint(failed_to_clear_bs, L_left_prev_vec, cap_old_bs):
    """校验任务是否被拖延至第三帧"""
    if np.any(failed_to_clear_bs):
        bad_i, bad_j = np.where(failed_to_clear_bs)
        print("\n" + "!"*60)
        print(f"[FATAL ERROR] 跨帧假设打破：基站本帧算力无法清空历史残余！")
        print(f"案发坐标: 基站 {bad_i[0]}, 用户 {bad_j[0]}")
        print(f"旧债: {L_left_prev_vec[bad_i[0], bad_j[0]]:.2f} bits, 本帧最大处理量: {cap_old_bs[bad_i[0], bad_j[0]]:.2f} bits")
        print("!"*60)
        raise RuntimeError("Task span exceeded two frames: Failed to clear L_left_prev_bs")
