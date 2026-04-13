import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SystemConfig


class OffloadingActor(nn.Module):
    """
    论文中的 DNN 模型 (Actor)，用于生成卸载决策概率。
    对应 Algorithm 1, Line 6 以及 Fig. 3 中的 DNN 结构。
    """

    def __init__(self, num_ues, hidden_dim=256):
        super(OffloadingActor, self).__init__()

        # --- 1. 确定输入维度 ---
        # 根据论文 Section IV-B  State X_{t,i} 包含:
        # 1. Q_ij(t): J 个用户的任务队列长度 -> J
        # 2. E_i(t): 基站当前的能量队列 -> 1
        # 3. T^{BS, left}_{t-1}: 基站剩余处理时间 -> 1
        # 4. R^{BS}_{t,ij}: J 个用户到 BS 的速率 -> J
        # 5. R^{S}_{t,ij}: J 个用户到 LEOS 的速率 -> J
        # 总维度 = 3 * J + 2
        self.input_dim = 3 * num_ues + 2
        self.output_dim = num_ues  # 输出每个用户的卸载概率 (J 维)

        # --- 2. 定义网络层 ---
        # --- 新增功能：自动化输入归一化层 ---
        # 这一层放在最前面，专门负责维护输入 X 的均值和方差
        # affine=False 表示我们只做归一化，不需要学习缩放参数 gamma 和 beta
        # momentum=0.01 表示更新速度，类似滑动平均的 alpha，0.01 意味着它会很平滑地记录长期统计
        self.input_bn = nn.BatchNorm1d(self.input_dim, affine=False, momentum=0.01)

        # 后续的全连接层
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 隐藏层的 BN

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

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
                        nn.init.constant_(m.bias, 6.0)
                    else:
                        nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        state: (batch_size, input_dim)
        """
        # 1. 自动化归一化
        # 这里会自动应用维护好的 running_mean 和 running_var
        x = self.input_bn(state)

        # 2. 常规网络传播
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        logits = self.output_layer(x)
        prob = torch.sigmoid(logits)

        return prob

class FocalLoss(nn.Module):
    """
    论文 Eq.(45) 提到的 Focal Cross-Entropy Loss 。
    虽然公式 (45) 写的是标准 BCE，但文字描述为 "focal cross-entropy loss"。
    这里实现了带 gamma 参数的 Focal Loss，当 gamma=0 时退化为标准 BCE。
    """

    def __init__(self, alpha=0.5, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: DNN 输出的预测概率 \tilde{b}, shape (batch_size, J)
            targets: 最优决策 b^* (0 或 1), shape (batch_size, J)
        """
        # 防止 log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        # 计算标准 BCE (不带 reduction)
        # BCE = - [y * log(p) + (1-y) * log(1-p)]
        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))

        # 计算 pt (模型对正确类别的预测概率)
        # 如果 target=1, pt = p; 如果 target=0, pt = 1-p
        pt = torch.where(targets == 1, inputs, 1 - inputs)

        # Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Alpha Term (类别平衡，可选)，这里设置alpha_term = 1，就是先暂时不考虑这个因素的影响
        #alpha_term = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        alpha_term = 1
        loss = alpha_term * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_input_vector(Q, E, T_left, R_BS, R_LEOS):
    """
    获取多基站架构下的状态向量 (Multi-BS State Vector)
    输入:
        Q, R_BS, R_LEOS: shape (I, J)
        E, T_left: shape (I,)
    输出:
        torch.FloatTensor, shape (I, input_dim)
    """
    I, J = Q.shape
    state_list = []

    for i in range(I):
        # 1. 取出第 i 个基站局部的状态，并进行数量级缩放 (粗略归一化)
        scale_Q = Q[i] / 1e6  # shape: (J,)
        scale_E = np.array([E[i] / 10.0])  # 标量转为 shape: (1,)
        scale_T = np.array([T_left[i] / 1.0])  # 标量转为 shape: (1,)
        scale_R_BS = R_BS[i] /2e7  # shape: (J,)
        scale_R_LEOS = R_LEOS[i] / 1e7  # shape: (J,)

        # 2. 拼接第 i 个基站的一维状态向量，长度为 3*J + 2
        state_i = np.concatenate([
            scale_Q, scale_E, scale_T, scale_R_BS, scale_R_LEOS
        ])
        state_list.append(state_i)

    # 3. 将所有基站的状态打包成一个 Batch 送给 PyTorch
    return torch.FloatTensor(np.array(state_list))