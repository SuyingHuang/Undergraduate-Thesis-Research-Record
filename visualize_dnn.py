import sys, os
sys.path.insert(0, r"E:\PY_Project\LDA")

from config import SystemConfig
from core.models.dnn_model import OffloadingActor
from torchview import draw_graph
import torch

os.makedirs("results", exist_ok=True)

model = OffloadingActor(num_ues=10, hidden_dim=512)
model.eval()

x = torch.randn(1, model.input_dim)

g = draw_graph(
    model,
    input_data=[x],
    graph_name="OffloadingActor_DNN",
    expand_nested=True,
    save_graph=False,
)

# --- 后处理 DOT：改为网格布局，让图片接近正方形 ---
dot_source = g.visual_graph.source

# 1. 改流向：LR(单行) → TB(多行)
dot_source = dot_source.replace("rankdir=LR", "rankdir=TB")

# 2. 按逻辑阶段分组，每个阶段放在同一行 (rank=same)
# 节点编号对应关系（由 torchview 深度优先展开决定）：
#   0: 输入张量
#   1,2,3: InputProj (Linear → LayerNorm → relu)
#   4,5,6,7,8: ResidualBlock #1 (Linear→LN→Dropout→add→ReLU)
#   9,10,11,12,13: ResidualBlock #2
#   14,15,16: OutputProj (Linear → LayerNorm → relu)
#   17: Output Layer (Linear → sigmoid)
#   18: 输出张量
rank_groups = [
    [0],                    # 输入
    [1, 2, 3],              # 输入投影
    [4, 5, 6, 7, 8],       # 残差块 1
    [9, 10, 11, 12, 13],   # 残差块 2
    [14, 15, 16],           # 输出投影
    [17],                   # 输出层
    [18],                   # 输出
]

rank_lines = []
for group in rank_groups:
    nodes = "; ".join(str(n) for n in group)
    rank_lines.append(f"\t{{ rank=same; {nodes}; }}")

edge_marker = "\t0 -> 1"
insert_pos = dot_source.find(edge_marker)
if insert_pos >= 0:
    rank_block = "\n".join(rank_lines) + "\n"
    dot_source = dot_source[:insert_pos] + rank_block + dot_source[insert_pos:]

# 3. 调整尺寸与间距，适配正方形
dot_source = dot_source.replace(
    'graph [ordering=in rankdir=TB size="13.5,13.5"]',
    'graph [ordering=in rankdir=TB size="10,10" ranksep=0.6 nodesep=0.3]'
)

dot_path = "results/dnn_architecture.dot"
with open(dot_path, "w", encoding="utf-8") as f:
    f.write(dot_source)

print(f"DOT saved → {dot_path}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print("提示：把 .dot 文件拖到 https://dreampuf.github.io/GraphvizOnline 即可在线渲染为正方形图片")
