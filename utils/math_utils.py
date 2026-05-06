import numpy as np


def solve_cubic_newton(a, b, d, iterations=10):
    """
    稳健的牛顿迭代法求解: a*f^3 + b*f^2 + d = 0 (f >= 0)
    用于求解 Problem P3 中的 Type A 频率。
    """
    # 边界情况处理
    if abs(a) < 1e-25:
        if abs(b) < 1e-25: return 0.0
        val = -d / b
        return np.sqrt(val) if val > 0 else 0.0

    # 智能初始猜测: 取单项解的最小值作为起点
    x_cubic = (-d / a) ** (1 / 3)
    x_quad = np.sqrt(max(0, -d / b)) if b > 1e-20 else x_cubic
    x = min(x_cubic, x_quad)

    for _ in range(iterations):
        fx = a * x ** 3 + b * x ** 2 + d
        dfx = 3 * a * x ** 2 + 2 * b * x

        if abs(dfx) < 1e-15: break

        x_new = x - fx / dfx
        if abs(x_new - x) < 1e-6:
            return max(0.0, x_new)
        x = x_new

    return max(0.0, x)


def solve_cubic_newton_vectorized(a_arr, b_in, d_arr, iterations=10):
    """
    向量化牛顿迭代法，同时求解多用户的三次方程 a*f^3 + b*f^2 + d = 0。
    a_arr, d_arr 为形状 (N,) 的数组。
    b_in 可以是标量（所有用户共享）或形状 (N,) 的数组（每用户不同 lam/mu）。
    """
    n = len(a_arr)
    result = np.zeros(n)

    # 统一为数组
    b_arr = np.broadcast_to(np.asarray(b_in), (n,)).copy()

    # 边界情况: a≈0 退化为二次
    tiny_a = np.abs(a_arr) < 1e-25
    quad_only = tiny_a & (np.abs(b_arr) >= 1e-25)
    if np.any(quad_only):
        val = -d_arr[quad_only] / b_arr[quad_only]
        result[quad_only] = np.sqrt(np.maximum(0.0, val))

    # 正常三次情况
    active = ~tiny_a
    if not np.any(active):
        return result

    a = a_arr[active]
    b = b_arr[active]
    d = d_arr[active]
    m = len(a)

    # 初始猜测
    x_cubic = (-d / a) ** (1.0 / 3.0)
    b_large = b > 1e-20
    x_quad = np.where(b_large, np.sqrt(np.maximum(0.0, -d / b)), x_cubic)
    x = np.minimum(x_cubic, x_quad)

    converged = np.zeros(m, dtype=bool)

    for _ in range(iterations):
        x2 = x * x
        x3 = x2 * x
        fx = a * x3 + b * x2 + d
        dfx = 3.0 * a * x2 + 2.0 * b * x

        valid_step = (np.abs(dfx) > 1e-15) & ~converged
        if not np.any(valid_step):
            break

        x_new = np.where(valid_step, x - fx / np.maximum(np.abs(dfx), 1e-15), x)
        newly_converged = np.abs(x_new - x) < 1e-6
        converged = converged | newly_converged
        x = np.where(valid_step, x_new, x)

    result[active] = np.maximum(0.0, x)
    return result