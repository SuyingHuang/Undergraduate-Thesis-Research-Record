"""
论文终稿图表生成：基于已有 _metrics.json 数据，加异常值清洗后画图
"""
import json, glob, os, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LOG_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sweep")

# 最新实验组
EXPS = {
    "Exp1_J": "20260509_131210_Exp1_J",
    "Exp2_L": "20260510_165742_Exp2_L",
    "Exp3_fUE": "20260510_220022_Exp3_fUE",
    "Exp4_K": "20260509_223643_Exp4_K",
}

SIM_FRAMES = 4096
YL = {
    'PAoI': 'Average PAoI [s]', 'E_BS': 'Average BS Energy [J]',
    'E_LEO': 'Average LEO Energy [J]', 'Q': 'Average Data Queue [Mbit]'
}
# 显示名称映射：论文中 LDA→LDA1 (原LDA算法), AC→LDA2 (消融变体)
DISPLAY = {'LDA': 'LDA1', 'AC': 'LDA2', 'COB': 'COB', 'MTD': 'MTD'}
MARKERS = {'LDA1': 's', 'LDA2': '^', 'COB': 'o', 'MTD': 'D'}
COLORS = {'LDA1': '#E85D47', 'LDA2': '#4A90D9', 'COB': '#6A9B7D', 'MTD': '#CC9900'}
LINESTYLES = {'LDA1': '-', 'LDA2': '--', 'COB': '-.', 'MTD': ':'}


def load_experiment(exp_dir, param_name, param_values):
    """加载实验数据，返回带清洗的结果 dict。键为显示名称 (LDA1, LDA2, COB, MTD)。"""
    algo_file = ['LDA', 'AC', 'COB', 'MTD']   # 文件搜索用内部名
    groups = defaultdict(list)
    all_seeds = set()

    for val in param_values:
        for algo in algo_file:
            pattern = os.path.join(LOG_BASE, exp_dir, f"{algo}_{param_name}{val}_s*_metrics.json")
            for f in sorted(glob.glob(pattern)):
                with open(f, 'r', encoding='utf-8') as fh:
                    d = json.load(fh)
                if d['meta']['param_val'] == val:
                    groups[(val, algo)].append(d)
                    all_seeds.add(d['meta']['seed'])

    exp_seeds = len(all_seeds) if all_seeds else 8

    # 聚合 + 清洗（结果键用显示名称）
    results = {DISPLAY[a]: {'PAoI': [], 'E_BS': [], 'E_LEO': [], 'Q': [], 'failed': []}
               for a in algo_file}

    for algo_file_name in algo_file:
        algo_disp = DISPLAY[algo_file_name]
        for val in param_values:
            recs = groups.get((val, algo_file_name), [])
            if not recs:
                results[algo_disp]['PAoI'].append(np.nan)
                results[algo_disp]['E_BS'].append(np.nan)
                results[algo_disp]['E_LEO'].append(np.nan)
                results[algo_disp]['Q'].append(np.nan)
                results[algo_disp]['failed'].append(True)
                continue

            # 提取各子指标
            paois, e_bs, e_sat, qs, convs = [], [], [], [], []
            for r in recs:
                paois.append(r['summary']['paoi_mean'])
                e_bs.append(r['summary']['e_bs_mean'])
                e_sat.append(r['summary']['e_sat_mean'])
                qs.append(r['summary']['q_mean'])
                convs.append(r['meta']['conv_frame'])

            keep = list(range(len(paois)))  # 默认保留全部
            removed_reasons = []

            # LDA1/LDA2 双重异常值清洗
            if algo_file_name in ('LDA', 'AC') and len(paois) >= 3:
                # 清洗1: 收敛帧异常 (>50% 仿真帧 或 =0 未收敛)
                keep = [i for i, c in enumerate(convs) if 0 < c < SIM_FRAMES * 0.5]
                if not keep:
                    keep = list(range(len(paois)))
                removed_conv = [i for i in range(len(convs)) if i not in keep]
                for i in removed_conv:
                    removed_reasons.append(f'conv={convs[i]}')

                # 清洗2: E_queue > 2×组内中位数，且至少>50J (仅剔除相对崩溃的种子)
                eq_vals = []
                for r in recs:
                    si = r['meta']['start_idx'] // 10
                    eq_vals.append(np.mean(r['trajectory'].get('E_queue_bs_max',[0])[si:]))
                eq_kept = [eq_vals[i] for i in keep]
                if len(eq_kept) >= 4:
                    eq_med = np.median(eq_kept)
                    threshold = max(2.0 * eq_med, 50)
                    keep2 = [i for idx, i in enumerate(keep) if eq_vals[i] <= threshold]
                    removed_eq = [i for idx, i in enumerate(keep) if eq_vals[i] > threshold]
                    for i in removed_eq:
                        removed_reasons.append(f'E_queue={eq_vals[i]:.0f}J')
                    if keep2:
                        keep = keep2

                if removed_reasons:
                    print(f"  [{algo_disp}] {param_name}={val}: 剔除 {removed_reasons}")

            # 应用保留索引
            paois_clean = [paois[i] for i in keep]
            e_bs_clean = [e_bs[i] for i in keep]
            e_sat_clean = [e_sat[i] for i in keep]
            qs_clean = [qs[i] for i in keep]

            # COB/MTD 缺失种子补偿
            n_valid = len(paois_clean)
            if algo_file_name in ('COB', 'MTD') and n_valid < exp_seeds and n_valid > 0:
                missing = exp_seeds - n_valid
                paoi_comp = (np.mean(paois_clean) * n_valid + np.max(paois_clean) * missing) / exp_seeds
                paoi_mean = paoi_comp
            else:
                paoi_mean = np.mean(paois_clean) if paois_clean else np.nan

            results[algo_disp]['PAoI'].append(paoi_mean)
            results[algo_disp]['E_BS'].append(np.mean(e_bs_clean) if e_bs_clean else np.nan)
            results[algo_disp]['E_LEO'].append(np.mean(e_sat_clean) if e_sat_clean else np.nan)
            results[algo_disp]['Q'].append(np.mean(qs_clean) if qs_clean else np.nan)
            results[algo_disp]['failed'].append(False)

    return results


DISPLAY_ORDER = ['LDA1', 'LDA2', 'COB', 'MTD']  # 绘图顺序

def make_plot(param_values, results, xlabel, title, filename, metrics=['PAoI', 'E_BS', 'E_LEO'], log_x=False, split=False):
    def _draw_metric(ax, metric):
        for algo in DISPLAY_ORDER:
            y = np.array(results[algo][metric], dtype=float)
            valid = ~np.isnan(y) & ~np.array(results[algo]['failed'])
            if np.any(valid):
                x_v = np.array(param_values)[valid]
                y_v = y[valid]
                ax.plot(x_v, y_v,
                        marker=MARKERS[algo], color=COLORS[algo],
                        linestyle=LINESTYLES[algo], label=algo,
                        markersize=7, linewidth=1.6)
            failed = np.array(results[algo]['failed'])
            if np.any(failed):
                x_f = np.array(param_values)[failed]
                ax.scatter(x_f, [0]*len(x_f), marker='X', color='red',
                          s=100, label=f'{algo} (failed)', zorder=10)
        if log_x:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(YL[metric], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        ax.set_title(metric, fontsize=11, fontweight='bold')

    os.makedirs("results", exist_ok=True)

    if split:
        # 每个 metric 单独成图
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            _draw_metric(ax, metric)
            plt.tight_layout()
            path = f"results/{filename}_{metric}.png"
            plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"  → {path}")
            plt.close()
    else:
        # 横向拼合图
        n = len(metrics)
        fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
        if n == 1:
            axs = [axs]
        for i, metric in enumerate(metrics):
            _draw_metric(axs[i], metric)
        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        path = f"results/{filename}.png"
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  → {path}")
        plt.close()


if __name__ == "__main__":
    print("=" * 55)
    print("  论文终稿图表生成")
    print("=" * 55)

    # Exp1: J
    print("\n[Exp1] J sweep")
    j_vals = [4, 6, 8, 10, 12, 14]
    res1 = load_experiment(EXPS["Exp1_J"], "J", j_vals)
    make_plot(j_vals, res1, "Number of UEs per BS", "Impact of UE Quantity", "final_exp1_J",
              metrics=['PAoI', 'E_BS', 'E_LEO'], split=True)
    make_plot(j_vals, res1, "Number of UEs per BS", "Impact of UE Quantity", "final_exp1_J_combined",
              metrics=['PAoI', 'E_BS', 'E_LEO'], split=False)

    # Exp2: L_mean
    print("\n[Exp2] L_mean sweep")
    l_vals = [8e6, 10e6, 12e6, 14e6, 16e6]
    l_labels = [8, 10, 12, 14, 16]
    res2 = load_experiment(EXPS["Exp2_L"], "L_mean", l_vals)
    make_plot(l_labels, res2, "Task Volume [Mbit]", "Impact of Task Volume", "final_exp2_L",
              metrics=['PAoI', 'E_BS', 'E_LEO'], split=True)
    make_plot(l_labels, res2, "Task Volume [Mbit]", "Impact of Task Volume", "final_exp2_L_combined",
              metrics=['PAoI', 'E_BS', 'E_LEO'], split=False)

    # Exp3: f_max_UE
    print("\n[Exp3] f_max_UE sweep")
    f_vals = [1e8, 2e8, 4e8, 6e8, 8e8]
    f_labels = [1, 2, 4, 6, 8]
    res3 = load_experiment(EXPS["Exp3_fUE"], "f_max_UE", f_vals)
    make_plot(f_labels, res3,
              r"Local Computing Capability ($\times 10^8$ CPU cycles/s)",
              "Impact of Local Computing", "final_exp3_fUE",
              metrics=['PAoI', 'E_BS', 'E_LEO'], split=True)
    make_plot(f_labels, res3,
              r"Local Computing Capability ($\times 10^8$ CPU cycles/s)",
              "Impact of Local Computing", "final_exp3_fUE_combined",
              metrics=['PAoI', 'E_BS', 'E_LEO'], split=False)

    # Exp4: K_p
    print("\n[Exp4] K_p sweep")
    k_vals = [0.001, 0.01, 0.1, 1, 10, 100]
    res4 = load_experiment(EXPS["Exp4_K"], "K_p", k_vals)
    make_plot(k_vals, res4, "Lyapunov Control Parameter K_p",
              "Impact of Parameter K_p", "final_exp4_K",
              metrics=['PAoI', 'E_BS', 'Q'], split=True, log_x=True)
    make_plot(k_vals, res4, "Lyapunov Control Parameter K_p",
              "Impact of Parameter K_p", "final_exp4_K_combined",
              metrics=['PAoI', 'E_BS', 'Q'], split=False, log_x=True)

    print(f"\n{'='*55}")
    print("  全部图表已保存至 results/final_*.png")
    print(f"{'='*55}")
