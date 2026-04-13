import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1, iv # iv 是第一类修正贝塞尔函数，用于 Rician
import matplotlib.style as style

# 设置绘图风格，使其更接近学术期刊
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
style.use('seaborn-v0_8-muted')

# ============================================================
# 1. 核心 PDF 函数定义
# ============================================================
def shadowed_rician_pdf(x, b_k, Omega_k, m_k):
    """Shadowed-Rician 信道增益平方 |h|^2 的 PDF"""
    x = np.maximum(x, 1e-9)
    two_b = 2 * b_k
    term1 = ((two_b * m_k) / (two_b * m_k + Omega_k)) ** m_k
    term2 = (1 / two_b) * np.exp(-x / two_b)
    hyp_arg = (Omega_k * x) / (two_b * (two_b * m_k + Omega_k))
    term3 = hyp1f1(m_k, 1, hyp_arg)
    return np.nan_to_num(term1 * term2 * term3)

def rician_power_pdf(x, b_k, Omega_k):
    """纯 Rician 信道增益平方 |h|^2 的 PDF (作为对比基准)"""
    # 在功率域，Rician 变为非中心卡方分布的变体
    # 公式: f(x) = 1/(2b) * exp(-(x+Omega)/(2b)) * I0(sqrt(x*Omega)/b)
    two_b = 2 * b_k
    term1 = 1 / two_b
    term2 = np.exp(-(x + Omega_k) / two_b)
    term3 = iv(0, np.sqrt(x * Omega_k) / b_k)
    return term1 * term2 * term3

# ============================================================
# 2. 参数配置
# ============================================================
b_k_base = 0.12  # 散射分量
Omega_k_base = 0.8 # 视距分量

shadowing_configs = {
    'Light Shadowing (m=20)':   {'b_k': 0.1,  'Omega_k': 0.8, 'm_k': 20,  'color': '#28A745'},
    'Average Shadowing (m=5)':  {'b_k': 0.12, 'Omega_k': 0.7, 'm_k': 5,   'color': '#FFC107'},
    'Heavy Shadowing (m=1)':    {'b_k': 0.15, 'Omega_k': 0.4, 'm_k': 1,   'color': '#DC3545'},
}

x = np.linspace(0, 5, 1000)

# ============================================================
# 3. Figure 1: PDF 演变与参考线对比
# ============================================================
plt.figure(figsize=(10, 6))

# A. 绘制 Shadowed-Rician 曲线
for label, p in shadowing_configs.items():
    pdf_vals = shadowed_rician_pdf(x, p['b_k'], p['Omega_k'], p['m_k'])
    plt.plot(x, pdf_vals, label=label, color=p['color'], lw=3)

# B. 绘制 Rician 参考线 (无遮蔽)
# 使用最轻遮蔽组的 b 和 Omega 作为基准
rician_ref = rician_power_pdf(x, 0.1, 0.8)
plt.plot(x, rician_ref, 'k--', lw=2, alpha=0.8, label='Pure Rician Ref (No Shadowing)')

# C. 绘制 Exponential 参考线 (Rayleigh 功率, 无视距)
exp_ref = (1/1.0) * np.exp(-x/1.0) # 假设总功率归一化为1
plt.plot(x, exp_ref, 'k:', lw=2, alpha=0.5, label='Exponential Ref (Rayleigh Power)')

plt.xlabel(r'信道增益平方 $|h_k|^2$', fontsize=12)
plt.ylabel(r'概率密度 $f_{|h|^2}(x)$', fontsize=12)
plt.title('图 1: Shadowed-Rician 概率密度演变\n(从 Rician 到 Rayleigh 的过渡)', fontsize=14)
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim([0, 4])
plt.ylim([0, 1.6])
plt.savefig('SR_PDF_Analysis.png', dpi=300)
plt.show()

# ============================================================
# 4. Figure 2: 统计特性统计
# ============================================================
plt.figure(figsize=(8, 6))

names = list(shadowing_configs.keys())
# 理论均值: E[|h|^2] = 2b + Omega
means = [2*p['b_k'] + p['Omega_k'] for p in shadowing_configs.values()]
# 理论方差: 描述信号抖动的剧烈程度
# 简化公式：Var = (2b)^2 + 2*Omega*(2b) + (Omega^2/m)*(m+1) - Omega^2
vars_val = [(2*p['b_k'])**2 + 2*p['Omega_k']*(2*p['b_k']) + (p['Omega_k']**2/p['m_k']) for p in shadowing_configs.values()]

x_pos = np.arange(len(names))
plt.bar(x_pos - 0.2, means, 0.4, label='均值 (信号强度)', color='#5DADE2', edgecolor='black', alpha=0.9)
plt.bar(x_pos + 0.2, vars_val, 0.4, label='方差 (衰落剧烈度)', color='#E74C3C', edgecolor='black', alpha=0.9)

plt.xticks(x_pos, names)
plt.ylabel('数值', fontsize=12)
plt.title('图 2: 不同遮蔽程度下的统计特性对比', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上方标注数值
for i, v in enumerate(vars_val):
    plt.text(i + 0.2, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('SR_Statistics.png', dpi=300)
plt.show()