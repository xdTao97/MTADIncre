import torch
import torch.nn as nn
from layer_module import *

class IncrementalModelWithGEM(nn.Module):
    def __init__(self, num_nodes, dim, out_dim, device, time_step, dynamic_nodes, memory_size=10, cout=16, heads=4,
                                          dropout=0.5, m=0.9, in_dim=256, is_add1=True):
        super(IncrementalModelWithGEM, self).__init__()
        self.num_nodes = num_nodes
        self.dynamic_nodes = dynamic_nodes
        self.time_step = time_step
        self.embed1 = nn.Embedding(dynamic_nodes, dim)
        self.m = m
        self.embed2 = nn.Embedding(dynamic_nodes, dim)
        self.m = m  # 动量因子
        self.device = device
        self.memory_data = []  # GEM记忆库
        self.memory_labels = []
        self.memory_size = memory_size  # 最大记忆容量
        self.dim = dim

        # 时间归一化模块
        self.time_norm = nn.LayerNorm(normalized_shape=[dim])
        # 融合模块
        self.gate_Fusion_1 = gatedFusion_1(dim, device)
        self.feat_expand = nn.Linear(self.dim, self.num_nodes)

    def add_to_memory(self, data, labels):
        """
        将增量样本添加到记忆库
        """
        self.memory_data.append(data)
        self.memory_labels.append(labels)
        if len(self.memory_data) > self.memory_size:
            self.memory_data.pop(0)
            self.memory_labels.pop(0)
        # 线性变换层，用于特征扩展

    def gem_update(self, nodevec_fusion):
        """
        使用 GEM 方法更新节点特征
         """
        # 扩展输入特征维度
        nodevec_fusion_expanded = self.feat_expand(nodevec_fusion)  # (256, 100, 100)
        if len(self.memory_data) > 0:
            memory_data = torch.cat(self.memory_data, dim=0)  # 合并记忆库数据
            memory_loss = torch.mean(
                (nodevec_fusion_expanded - memory_data) ** 2, dim=0
            )  # 计算与记忆库的特征差异
        else:
            memory_loss = torch.zeros_like(nodevec_fusion_expanded, device=self.device)

        return nodevec_fusion_expanded - memory_loss

    def forward(self, input, incre):
        """
        主 forward 方法，使用 GEM 进行增量学习
        """
        batch_size, nodes, time_step = input.shape[0], self.dynamic_nodes, self.time_step

        # 动量更新嵌入
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        if incre:
            # 分割数据为原始部分和增量部分
            part1, part2 = torch.split(input, [1, 54], dim=2)
            input = torch.cat([part1, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part1.shape[2],
                                                  device=self.device)], dim=2)
            part2_incre = torch.cat([part2, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part2.shape[2],
                                                        device=self.device)], dim=2)

            # 增量部分处理
            node_input_incre = part2_incre
            nodevec_fusion = self.time_norm(node_input_incre)
        else:
            nodevec_fusion = torch.zeros(input.shape[0], input.shape[1], input.shape[2], device=self.device)

        node_input = input + nodevec_fusion
        node_input = self.time_norm(node_input)

        idx = torch.arange(self.dynamic_nodes).to(self.device)
        nodevec_static = self.embed1(idx)

        # 节点融合
        nodevec_fusion = self.gate_Fusion_1(batch_size, nodevec_static, node_input) + nodevec_static

        # 使用 GEM 更新
        base_adj_gem = self.gem_update(nodevec_fusion)

        return base_adj_gem


