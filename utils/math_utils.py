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