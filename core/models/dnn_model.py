import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    等宽残差模块：Linear → LayerNorm → Dropout → 残差相加 → ReLU。
    """

    def __init__(self, dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = out + residual  # 残差连接
        out = self.activation(out)
        return out


class OffloadingActor(nn.Module):
    """
    论文中的 DNN 模型 (Actor)，用于生成卸载决策概率。
    对应 Algorithm 1, Line 6 以及 Fig. 3 中的 DNN 结构。
    采用 LayerNorm + 残差连接的结构。
    """

    def __init__(self, num_ues, num_bs=1, hidden_dim=640):
        super(OffloadingActor, self).__init__()

        # --- 1. 确定输入维度 ---
        # State X_{t,i} 包含:
        # 1. L_ij(t): J 个用户的当前任务量 -> J
        # 2. Q_bs_ij(t): 本BS各用户的独立BS积压 -> J
        # 3. Q_sat_ij(t): 本BS各用户的独立卫星积压 -> J
        # 4. E_i(t): 基站当前的能量队列 -> 1
        # 5. T^{BS, left}_{t-1}: 基站剩余处理时间 -> 1
        # 6. R^{BS}_{t,ij}: J 个用户到 BS 的速率 -> J
        # 7. R^{S}_{t,ij}: J 个用户到 LEOS 的速率 -> J
        # 8. 与当前联合决策有关的全局卫星竞争状态 -> 3 * I * J
        #    [可卸载任务量, 对应卫星积压, 假设选择卫星时的可用计算时间]
        # 旧卫星服务和能耗对同一帧所有候选均为常数，不作为 DNN 输入。
        self.input_dim = 5 * num_ues + 2 + 3 * num_bs * num_ues
        self.output_dim = num_ues  # 输出每个用户的卸载概率 (J 维)

        # --- 2. 定义网络层 ---
        # 输入层：线性 + LayerNorm
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)

        # 中间残差块
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_ln = nn.LayerNorm(hidden_dim // 2)

        self.output_layer = nn.Linear(hidden_dim // 2, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        """
        使用正交初始化 (Orthogonal Initialization) 替换 Kaiming 初始化
        """
        for name, m in self.named_modules():  # 注意这里改用了 named_modules 以便区分层名
            if isinstance(m, nn.Linear):
                # 1. 区分输出层和隐藏层
                if 'output_layer' in name:
                    # 输出层后面接的是 Sigmoid，通常使用标准增益 1.0
                    gain = nn.init.calculate_gain('sigmoid')
                else:
                    # 隐藏层后面接的是 ReLU
                    # 正交矩阵本身会保持方差不变，但 ReLU 会砍掉一半的负信号
                    # 所以必须乘上 sqrt(2) 来补偿丢失的能量
                    gain = nn.init.calculate_gain('relu')

                    # 2. 执行正交初始化
                nn.init.orthogonal_(m.weight, gain=gain)

                # 3. 偏置初始化
                if m.bias is not None:
                    if 'output_layer' in name:
                        # 输出层偏置设为正值，让网络初期倾向于 BS 卸载
                        # sigmoid(wx + b) 中，b>0 会使输出概率整体偏高
                        nn.init.constant_(m.bias, 1.0)
                    else:
                        nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        state: (batch_size, input_dim)
        Returns:
            logits: (batch_size, output_dim)，未经激活的原始输出
        """
        # 1. 输入投影 + LayerNorm
        x = self.input_proj(state)
        x = self.input_ln(x)
        x = F.relu(x)

        # 2. 残差块
        x = self.res_block1(x)
        x = self.res_block2(x)

        # 3. 输出投影
        x = self.output_proj(x)
        x = self.output_ln(x)
        x = F.relu(x)

        logits = self.output_layer(x)

        return logits

class FocalLoss(nn.Module):
    """
    带类别权重的 Focal BCE。gamma=0 时退化为加权 BCE；
    默认 alpha=0.5、mean reduction 得到逐元素 BCE 均值的 0.5 倍。

    输入为 logits，利用 binary_cross_entropy_with_logits 内部 LogSumExp 机制保证数值稳定性。
    """

    def __init__(self, alpha=0.5, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: DNN 输出的 logits, shape (batch_size, J)
            targets: 最优决策 b^* (0 或 1), shape (batch_size, J)
        """
        # 基础 BCE loss（不自己做 log/exp，由 LogSumExp 机制保证稳定）
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 内部局部做 sigmoid 计算 focal 权重
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        alpha_term = torch.where(
            targets == 1,
            torch.as_tensor(self.alpha, dtype=inputs.dtype, device=inputs.device),
            torch.as_tensor(1.0 - self.alpha, dtype=inputs.dtype, device=inputs.device)
        )
        loss = alpha_term * focal_term * bce_loss

        if mask is not None:
            mask = mask.to(dtype=loss.dtype, device=loss.device)
            loss = loss * mask
        if self.reduction == 'mean':
            if mask is None:
                return loss.mean()
            return loss.sum() / mask.sum().clamp_min(1.0)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_input_vector(L_t, Q_bs, Q_sat, E, T_left, R_BS, R_LEOS,
                     T_prop=None, offload_mask=None, tau=5.0):
    """
    获取多基站架构下的状态向量 (Multi-BS State Vector)
    输入:
        L_t: 当前帧任务量, shape (I, J)
        Q_bs: BS队列, shape (I, J)
        Q_sat: 卫星账本总积压（唯一来源）, shape (I, J)
        R_BS, R_LEOS, T_prop: shape (I, J)
        E, T_left: shape (I,)
        offload_mask: shape (I, J), True 表示任务不能在本地完成、需要卸载
    输出:
        torch.FloatTensor, shape (I, input_dim)
    """
    I, J = Q_bs.shape
    if tau <= 0:
        raise ValueError("tau must be positive")
    T_prop = (np.zeros_like(L_t, dtype=float) if T_prop is None
              else np.asarray(T_prop, dtype=float))
    offload_mask = (np.ones_like(L_t, dtype=bool) if offload_mask is None
                    else np.asarray(offload_mask, dtype=bool))
    if T_prop.shape != (I, J) or offload_mask.shape != (I, J):
        raise ValueError("T_prop and offload_mask must have shape (I, J)")

    # Candidate-independent old-satellite service is deliberately excluded.
    # The shared block describes only current tasks competing for the single
    # current satellite.  A zero rate yields zero usable compute time.
    sat_transfer = np.divide(
        L_t, R_LEOS,
        out=np.full((I, J), np.inf, dtype=float),
        where=np.asarray(R_LEOS) > 1e-9,
    )
    sat_available = np.maximum(0.0, tau - T_prop - sat_transfer)
    shared_sat = np.concatenate([
        np.where(offload_mask, L_t, 0.0).reshape(-1) / 1e6,
        np.where(offload_mask, Q_sat, 0.0).reshape(-1) / 1e6,
        np.where(offload_mask, sat_available, 0.0).reshape(-1) / tau,
    ])
    state_list = []

    for i in range(I):
        scale_L = L_t[i] / 1e6
        scale_Q_bs = Q_bs[i] / 1e6
        scale_Q_sat = Q_sat[i] / 1e6
        scale_E = np.array([E[i] / 10.0])
        scale_T = np.array([T_left[i] / 1.0])
        scale_R_BS = R_BS[i] / 2e7
        scale_R_LEOS = R_LEOS[i] / 1e7

        state_i = np.concatenate([
            scale_L, scale_Q_bs, scale_Q_sat, scale_E, scale_T,
            scale_R_BS, scale_R_LEOS, shared_sat
        ])
        state_list.append(state_i)

    return torch.FloatTensor(np.array(state_list))
