import torch
import torch.nn as nn
from layer_module import *

class IncrementalModelWithERGNN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device, time_step, dynamic_nodes,m=0.9):
        super(IncrementalModelWithERGNN, self).__init__()

        self.dynamic_nodes = dynamic_nodes
        self.time_step = time_step
        self.device = device
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        # Embedding layers for static and dynamic embeddings
        self.m=m
        self.embed1 = nn.Embedding(dynamic_nodes, in_dim)
        self.embed2 = nn.Embedding(dynamic_nodes, in_dim)

        # Time normalization layer
        self.time_norm = nn.LayerNorm(in_dim)

        # Node fusion module
        # 时间归一化模块
        self.time_norm = nn.LayerNorm(normalized_shape=[in_dim])
        # 融合模块
        self.gate_Fusion_1 = gatedFusion_1(in_dim, device)
        self.feat_expand = nn.Linear(self.in_dim, self.num_nodes)

        # Graph learning module
        self.graph_learn = nn.Sequential(
            nn.Linear(self.num_nodes, self.num_nodes),
            nn.ReLU(),
            nn.Linear(self.num_nodes, self.num_nodes),
            nn.Sigmoid()
        )

    def forward(self, input, incre):
        batch_size, nodes, time_step = input.shape[0], self.dynamic_nodes, self.time_step

        # Momentum update for dynamic embeddings
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        if incre:
            # Split input into raw and incremental datasets
            part1, part2 = torch.split(input, [24, 1], dim=2)

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
        # Graph learning: Compute dynamic adjacency matrix
        nodevec_fusion_expanded = self.feat_expand(nodevec_fusion)  # (256, 100, 100)
        base_adj = self.graph_learn(nodevec_fusion_expanded)

        return base_adj


