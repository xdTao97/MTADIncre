import torch
import torch.nn as nn
import torch.nn.functional as F
from layer_module import *

class MSLIncre(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device, time_step, dynamic_nodes, m, heads=4):
        super(MSLIncre, self).__init__()
        self.dynamic_nodes = dynamic_nodes
        self.time_step = time_step
        self.device = device
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        # 模型组件
        self.embed1 = nn.Embedding(dynamic_nodes, in_dim)
        self.embed2 = nn.Embedding(dynamic_nodes, in_dim)
        self.time_norm = nn.LayerNorm(time_step).to(device)  # 时间归一化
        self.gate_Fusion_1 = nn.Linear(in_dim, in_dim).to(device)  # 节点特征融合
        self.graph_update = nn.Linear(in_dim,  self.num_nodes).to(device)  # 图特征更新
        self.m = m  # 动量参数
        # Node fusion module
        # 时间归一化模块
        self.time_norm = nn.LayerNorm(normalized_shape=[in_dim])
        # 融合模块
        self.gate_Fusion_1 = gatedFusion_1(in_dim, device)
        self.feat_expand = nn.Linear(self.in_dim, self.num_nodes)
        # SSM 模块
        self.ssm_state_transition = nn.Linear(in_dim, in_dim).to(device)  # 状态转移矩阵
        self.ssm_observation = nn.Linear(in_dim, in_dim).to(device)  # 观测矩阵

    def graph_learn(self, nodevec_fusion):
        """
        基于节点特征动态生成邻接矩阵
        """
        adj_matrix = torch.matmul(nodevec_fusion, nodevec_fusion.transpose(1, 2))
        adj_matrix = torch.sigmoid(adj_matrix)  # 将值归一化到 [0, 1]
        return adj_matrix

    def msl_forward(self, input_features):

        state = torch.zeros(input_features.shape[0], self.in_dim, device=self.device)  # 初始化状态
        output_features = []

        for t in range(input_features.shape[2]):  # 时间步处理
            observation = input_features[:, :, t]
            state = torch.tanh(self.ssm_state_transition(state) + self.ssm_observation(observation))
            output_features.append(state.unsqueeze(2))  # 添加时间步特征

        return torch.cat(output_features, dim=2)

    def forward(self, input, incre):
        batch_size, nodes, time_step = input.shape[0], self.dynamic_nodes, self.time_step

        # Momentum update for dynamic embeddings
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        if incre:
            # Split input into raw and incremental datasets
            part1, part2 = torch.split(input, [54, 1], dim=2)

            # Pad the sequences to match the original input shape
            input = torch.cat([part1, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part1.shape[2],
                                                  device=self.device)], dim=2)
            part2_incre = torch.cat([part2, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part2.shape[2],
                                                        device=self.device)], dim=2)

            # Incremental processing: Generate dynamic features and adjust graph learning results
            node_input_incre = part2_incre
            nodevec_fusion = self.time_norm(node_input_incre)
        else:
            nodevec_fusion = torch.zeros(input.shape[0], input.shape[1], input.shape[2], device=self.device)

        # Combine static and dynamic node features
        node_input = input + nodevec_fusion
        node_input = self.time_norm(node_input)

        # Static embeddings
        idx = torch.arange(self.dynamic_nodes).to(self.device)
        nodevec_static = self.embed1(idx)


        nodevec_fusion = torch.tanh(self.gate_Fusion_1(batch_size, nodevec_static, node_input)) + nodevec_static

        # 动态生成邻接矩阵
        adj_matrix = self.graph_learn(nodevec_fusion)

        # 更新节点特征
        updated_features = self.graph_update(adj_matrix @ nodevec_fusion)

        return updated_features
