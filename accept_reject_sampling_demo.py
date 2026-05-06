import numpy as np
import matplotlib.pyplot as plt

# ===== 中文字体设置（很关键）=====
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 常用
plt.rcParams['axes.unicode_minus'] = False

# ===== 定义分布 =====
def p(x):
    return np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(5 * x))

def q(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

# ===== 生成数据 =====
x = np.linspace(-3, 3, 2000)

# ===== 自动计算 M =====
M = np.max(p(x) / q(x)) * 1.05  # 留一点余量

# ===== 绘图 =====
plt.figure()

plt.plot(x, p(x), label="目标分布 p(x)")
plt.plot(x, q(x), label="提议分布 q(x)")
plt.plot(x, M * q(x), linestyle='--', label="包络分布 Mq(x)")

plt.title("接受-拒绝采样示意图")
plt.xlabel("自变量 x")
plt.ylabel("密度")

plt.legend()
plt.grid(alpha=0.3)

plt.show()