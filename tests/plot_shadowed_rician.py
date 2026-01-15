import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1

# --- 1. 系统参数设置 (引用自论文 Table II) ---
I, J = 3, 10
B_Ka = 1e9  # 1 GHz
f_Ka = 30e9
lambda_Ka = 3e8 / f_Ka
sigma2_noise_linear = 1.37e-18
beta = 2.2
P_dBm = 33.0  # 发射功率 待定？
G_t_dBi = 42  # 发射天线增益 待定？
G_r_dBi = 44  # 接收天线增益 待定？

# Shadowed-Rician 参数
b_k, Omega_k, m_k = 0.158, 1.29, 20


# --- 2. 核心数学模型 ---
def target_pdf(x):
    """ 公式 (5): |h_k|^2 的理论概率密度函数 """
    two_b = 2 * b_k
    term1 = ((two_b * m_k) / (two_b * m_k + Omega_k)) ** m_k
    term2 = (1 / two_b) * np.exp(-x / two_b)
    hyp_arg = (Omega_k * x) / (two_b * (two_b * m_k + Omega_k))
    term3 = hyp1f1(m_k, 1, hyp_arg)
    return term1 * term2 * term3

def get_samples(n):
    """ 接受-拒绝采样逻辑 """
    x_max = 8
    # 自动确定包络高度 f_max
    x_test = np.linspace(0, x_max, 1000)
    f_max = np.max([target_pdf(i) for i in x_test]) * 1.1

    samples_sq = []
    while len(samples_sq) < n:
        x_c = np.random.uniform(0, x_max)
        if np.random.rand() < target_pdf(x_c) / f_max:
            samples_sq.append(x_c)
    return np.array(samples_sq)

def calculate_rate(dist, h_sq_val):
    """ 对数域链路预算计算速率 """
    # 自由空间路径损耗 (20 log10)
    PL_dB = 20* np.log10(4 * np.pi / lambda_Ka)
    PL_beta = 10*beta*np.log10(dist);
    # 衰落增益 (10 log10 因为输入是平方值)
    fading_dB = 10 * np.log10(h_sq_val)
    # 接收功率
    Pr_dBm = P_dBm + G_t_dBi + G_r_dBi + fading_dB - PL_dB-PL_beta
    # 转回线性 W
    Pr_linear = 10 ** ((Pr_dBm - 30) / 10)
    # 香农公式
    snr = Pr_linear / sigma2_noise_linear
    return (B_Ka / (I * J)) * np.log2(1 + snr)


# --- 3. 生成数据 ---
N = 10000
h_sq_samples = get_samples(N)
h_sq_max = np.max(h_sq_samples)
h_sq_mean = np.mean(h_sq_samples)

# --- 4. 绘图 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图 1: PDF (概率密度分布)
x_pdf = np.linspace(0, 8, 500)
y_pdf = [target_pdf(i) for i in x_pdf]
axes[0].hist(h_sq_samples, bins=50, density=True, alpha=0.5, color='gray', label='采样样本直方图')
axes[0].plot(x_pdf, y_pdf, 'r-', lw=2, label='理论 PDF (公式 5)')
axes[0].set_title("信道增益平方 $|h_k|^2$ 的 PDF")
axes[0].set_xlabel("$|h_k|^2$")
axes[0].set_ylabel("概率密度")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图 2: CDF (累积概率分布)
sorted_sq = np.sort(h_sq_samples)
cdf = np.arange(len(sorted_sq)) / float(len(sorted_sq))
axes[1].plot(sorted_sq, cdf, color='blue', lw=2)
axes[1].set_title("信道增益平方 $|h_k|^2$ 的 CDF")
axes[1].set_xlabel("$|h_k|^2$")
axes[1].set_ylabel("累积概率")
axes[1].grid(True, alpha=0.3)

# 图 3: 速率随距离变化
dists = np.linspace(200e3, 300e3, 100)
rate_peak = [calculate_rate(d, h_sq_max) / 1e6 for d in dists]
rate_avg = [calculate_rate(d, h_sq_mean) / 1e6 for d in dists]
axes[2].plot(dists / 1e3, rate_peak, 'r-', label='峰值速率 ($\max|h_k|^2$)')
axes[2].plot(dists / 1e3, rate_avg, 'g--', label='平均速率 ($E[|h_k|^2]$)')
axes[2].set_title("传输速率 vs. 距离")
axes[2].set_xlabel("距离 (km)")
axes[2].set_ylabel("速率 (Mbps)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_simulation.png')
plt.show()