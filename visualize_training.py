"""
经验回放与训练机制可视化
用于《DNN设计与训练过程》论文
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,  # 防止负号显示异常
})

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

# ============================================================
# 配色
# ============================================================
C_STORAGE = "#4A90D9"      # 存储路径 (蓝)
C_TRAINING = "#E85D47"     # 训练路径 (橙红)
C_BUFFER = "#F5F0E8"       # Replay Buffer 背景
C_SAMPLE = "#FFD166"       # 被采样格子
C_FRAME_BOX = "#E8F4FD"    # 帧处理背景
C_TRAIN_BOX = "#FDEFEF"    # 训练背景
C_DELTA = "#6A9B7D"        # delta 自适应

# ============================================================
# 1. 左侧：逐帧存储 (Frame-level Storage)
# ============================================================
ax.text(2.2, 7.6, "逐帧存储 (每帧执行)", ha="center", fontsize=12, fontweight="bold", color=C_STORAGE)

# 帧处理流水线 (画 3 帧示意)
for idx, y in enumerate([6.8, 6.1, 5.4]):
    # 帧标签
    ax.text(0.3, y + 0.2, f"Frame\nt={idx}", ha="center", fontsize=8, fontweight="bold", va="center")

    boxes = ["Env\nState", "DNN\nForward", "Action\n(b*)", "→ Store"]
    box_w, box_h = 0.65, 0.55
    for j, label in enumerate(boxes):
        x0 = 0.7 + j * (box_w + 0.08)
        color = C_FRAME_BOX if j < 3 else "#D4E6F9"
        rect = FancyBboxPatch((x0, y), box_w, box_h,
                              boxstyle="round,pad=0.05", facecolor=color,
                              edgecolor=C_STORAGE, linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x0 + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=7.5)

    # 箭头
    for j in range(3):
        x_start = 0.7 + j * (box_w + 0.08) + box_w
        x_end = 0.7 + (j + 1) * (box_w + 0.08)
        ax.annotate("", xy=(x_end, y + box_h / 2), xytext=(x_start, y + box_h / 2),
                    arrowprops=dict(arrowstyle="->", color=C_STORAGE, lw=1.2))

    # 最后一个箭头到右侧 Buffer
    ax.annotate("", xy=(4.8, y + box_h / 2),
                xytext=(0.7 + 3 * (box_w + 0.08) + box_w, y + box_h / 2),
                arrowprops=dict(arrowstyle="->", color=C_STORAGE, lw=1.2, connectionstyle="arc3,rad=0.15"))

# 存储箭头标注
ax.annotate("写入\n(FIFO)", xy=(4.2, 5.0), fontsize=7.5, color=C_STORAGE, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C_STORAGE, alpha=0.8))

# ============================================================
# 2. 中间：Replay Buffer (环形队列)
# ============================================================
# Buffer 网格
buf_cols, buf_rows = 16, 4  # 64 格展示 (实际 1024)
buf_x0, buf_y0 = 5.3, 6.55
cell_w, cell_h = 0.23, 0.23

# 大外框
outer = FancyBboxPatch((buf_x0 - 0.15, buf_y0 - 0.5), buf_cols * cell_w + 0.3,
                       buf_rows * cell_h + 0.8,
                       boxstyle="round,pad=0.1", facecolor="#F8F6F2",
                       edgecolor="#B0A890", linewidth=1.5)
ax.add_patch(outer)
ax.text(buf_x0 + buf_cols * cell_w / 2, buf_y0 - 0.3, "Replay Buffer (capacity=1024)",
        ha="center", fontsize=9, fontweight="bold", color="#5C5540")

# 画格子
np.random.seed(42)
sampled_cells = set()
while len(sampled_cells) < 8:  # 8 个高亮表示采样
    sampled_cells.add((np.random.randint(0, buf_rows), np.random.randint(0, buf_cols)))

# 写入指针位置
write_ptr_col = 7  # 当前写入位置

for r in range(buf_rows):
    for c in range(buf_cols):
        xc = buf_x0 + c * cell_w
        yc = buf_y0 - r * cell_h
        if (r, c) in sampled_cells:
            fc = C_SAMPLE
            ec = "#CC9900"
        elif c == write_ptr_col and r == 0:
            fc = "#E0F0FF"
            ec = C_STORAGE
        else:
            fc = "#FFFFFF"
            ec = "#D0CEC8"
        rect = FancyBboxPatch((xc, yc), cell_w - 0.02, cell_h - 0.02,
                              boxstyle="round,pad=0.01", facecolor=fc,
                              edgecolor=ec, linewidth=0.6)
        ax.add_patch(rect)

# 标注采样格
for (r, c) in list(sampled_cells)[:3]:
    xc = buf_x0 + c * cell_w + cell_w / 2
    yc = buf_y0 - r * cell_h - 0.12
    ax.plot(xc, yc, marker="*", color=C_TRAINING, markersize=8, zorder=10)

# 写入指针箭头
wp_x = buf_x0 + write_ptr_col * cell_w + cell_w / 2
wp_y = buf_y0 + 0.15
ax.annotate("写入指针", xy=(wp_x, buf_y0 + 0.08), xytext=(wp_x, buf_y0 + 0.5),
            ha="center", fontsize=7, color=C_STORAGE,
            arrowprops=dict(arrowstyle="->", color=C_STORAGE, lw=0.8))

# ============================================================
# 3. 右侧：定时训练 (Per-train_interval)
# ============================================================
ax.text(12.2, 7.6, "定时训练 (每 10 帧触发)", ha="center", fontsize=12, fontweight="bold", color=C_TRAINING)

# 从 Buffer 到训练的采样箭头
ax.annotate("", xy=(11.5, 5.5), xytext=(buf_x0 + buf_cols * cell_w + 0.15, 5.5),
            arrowprops=dict(arrowstyle="->", color=C_TRAINING, lw=2.0, connectionstyle="arc3,rad=-0.1"))
ax.text(10.6, 5.2, "随机采样\nbatch_size=64", ha="center", fontsize=8, color=C_TRAINING,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C_TRAINING, alpha=0.8))

# 训练流程
train_y = 4.2
train_steps = [
    ("State\nBatch", C_FRAME_BOX, C_STORAGE),
    ("DNN\nForward", C_FRAME_BOX, C_STORAGE),
    ("Logits\n+ Sigmoid", C_FRAME_BOX, C_STORAGE),
    ("FocalLoss\n(BCE)", "#FDE8E8", C_TRAINING),
    ("Backward\n+ Update", "#FDE8E8", C_TRAINING),
    ("delta_t\nAdapt", "#E4F0E8", C_DELTA),
]
box_w2, box_h2 = 1.05, 0.65
train_x0 = 11.0
for j, (label, fc, ec) in enumerate(train_steps):
    x0 = train_x0 + j * (box_w2 + 0.1)
    y0 = train_y
    rect = FancyBboxPatch((x0, y0), box_w2, box_h2,
                          boxstyle="round,pad=0.05", facecolor=fc,
                          edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x0 + box_w2 / 2, y0 + box_h2 / 2, label, ha="center", va="center", fontsize=7.5)

    if j < len(train_steps) - 1:
        ax.annotate("", xy=(x0 + box_w2 + 0.08, y0 + box_h2 / 2),
                    xytext=(x0 + box_w2, y0 + box_h2 / 2),
                    arrowprops=dict(arrowstyle="->", color=ec, lw=1.0))

# 虚线框标注 "训练仅在 Frame % 10 == 0 时执行"
train_bbox = FancyBboxPatch((train_x0 - 0.15, train_y - 0.25),
                            len(train_steps) * (box_w2 + 0.1) - 0.1 + 0.3, box_h2 + 0.5,
                            boxstyle="round,pad=0.1", facecolor="none",
                            edgecolor=C_TRAINING, linewidth=1.5, linestyle="--")
ax.add_patch(train_bbox)
ax.text(train_x0 + len(train_steps) * (box_w2 + 0.1) / 2 - 0.05, train_y + box_h2 + 0.55,
        "仅每 10 帧触发  |  使用每个 BS 独立的 Replay Buffer",
        ha="center", fontsize=8, color=C_TRAINING, fontstyle="italic")

# ============================================================
# 4. 右下角：delta_t 自适应曲线
# ============================================================
delta_ax = ax.inset_axes([0.78, 0.08, 0.2, 0.18])
frames = np.arange(0, 500)
train_frames = frames[frames % 10 == 0]
delta_vals = []
d = 0.5
loss_ema = 0.3
loss_ema_slow = 0.3
for f in frames:
    if f in train_frames and f > 50:
        ratio = loss_ema / (loss_ema_slow + 1e-9)
        if ratio < 0.95:
            d = max(0.08, d * 0.995)
        elif ratio > 1.05:
            d = min(0.5, d * 1.008)
        loss_ema = 0.9 * loss_ema + 0.1 * (0.15 + 0.03 * np.sin(f / 80))
        loss_ema_slow = 0.99 * loss_ema_slow + 0.01 * (0.15 + 0.03 * np.sin(f / 80))
    delta_vals.append(d)

delta_ax.plot(frames, delta_vals, color=C_DELTA, lw=1.6)
delta_ax.fill_between(frames, 0.08, delta_vals, color=C_DELTA, alpha=0.15)
delta_ax.axhline(0.08, color="gray", lw=0.6, ls="--", alpha=0.6)
delta_ax.set_ylim(0.04, 0.55)
delta_ax.set_xlim(0, 500)
delta_ax.set_xlabel("Frame", fontsize=7)
delta_ax.set_ylabel(r"$\delta_t$", fontsize=8)
delta_ax.set_title(r"$\delta_t$ 自适应探索窗口", fontsize=8, fontweight="bold", color=C_DELTA)
delta_ax.tick_params(labelsize=6)

# ============================================================
# 5. 图例
# ============================================================
legend_elements = [
    mpatches.Patch(facecolor=C_FRAME_BOX, edgecolor=C_STORAGE, label="前向推理 (逐帧)"),
    mpatches.Patch(facecolor="#FDE8E8", edgecolor=C_TRAINING, label="训练 (每10帧)"),
    mpatches.Patch(facecolor=C_SAMPLE, edgecolor="#CC9900", label="被采样经验 (batch)"),
    mpatches.Patch(facecolor="#E4F0E8", edgecolor=C_DELTA, label=r"$\delta_t$ 自适应"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9,
          ncol=2, bbox_to_anchor=(0.02, 0.1))

# ============================================================
# 6. 时序指示
# ============================================================
# 底部帧刻度
for t in [0, 9, 10, 19, 20, 29, 30]:
    fcolor = C_TRAINING if t % 10 == 0 else C_STORAGE
    marker = "^" if t % 10 == 0 else "|"
    msize = 8 if t % 10 == 0 else 5
    ax.plot(0.5 + t * 0.25, 0.3, marker=marker, color=fcolor, markersize=msize, clip_on=False)
ax.text(0.5 + 15 * 0.25, 0.03, "帧序号 (Frame Index)", ha="center", fontsize=8, color="gray")
ax.text(0.5 + 0 * 0.25, -0.05, "存储触发点", ha="center", fontsize=6.5, color=C_STORAGE)
ax.text(0.5 + 10 * 0.25, -0.05, "训练触发点", ha="center", fontsize=6.5, color=C_TRAINING)

# 标题
fig.suptitle("DNN 经验回放与训练机制", fontsize=15, fontweight="bold", y=0.97)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig("results/experience_replay_training.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved → results/experience_replay_training.png")
