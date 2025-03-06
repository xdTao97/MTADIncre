import torch
import torch.nn.functional as F
from torch import nn
from layer_module import *

class IncrementalModelWithLwF(nn.Module):
    def __init__(self,  num_nodes, dim, out_dim, device, time_step, dynamic_nodes, memory_size=10, cout=16, heads=4,
                                          dropout=0.5, m=0.9, lwf_lambda=1.0):
        super(IncrementalModelWithLwF, self).__init__()
        self.dynamic_nodes = dynamic_nodes
        self.time_step = time_step
        self.device = device
        self.m = m  # Momentum
        self.lwf_lambda = lwf_lambda  # LwF loss权重
        self.dim = dim
        # 模型组件
        self.embed1 = nn.Embedding(dynamic_nodes, dim)
        self.embed2 = nn.Embedding(dynamic_nodes, dim)
        # 时间归一化模块
        self.time_norm = nn.LayerNorm(normalized_shape=[dim])
        self.gate_Fusion_1 = gatedFusion_1(dim, device)
        self.graph_learn = nn.Linear(100, 100).to(device)

        # 保存旧模型的输出
        self.old_model_output = None

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, input, incre):
        batch_size, nodes, time_step = input.shape[0], self.dynamic_nodes, self.time_step

        # Momentum update for embeddings
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        base_adj_incre = None
        if incre:
            # Split datasets into raw and incremental datasets
            part1, part2 = torch.split(input, [1, 24], dim=2)
            input = torch.cat([part1, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part1.shape[2],
                                                  device=self.device)], dim=2)
            part2_incre = torch.cat([part2, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part2.shape[2],
                                                        device=self.device)], dim=2)
            # 增量处理：动态生成额外特征
            node_input_incre = part2_incre
            nodevec_fusion = self.time_norm(node_input_incre)
        else:
            nodevec_fusion = torch.zeros(input.shape[0], input.shape[1], input.shape[2]).to(self.device)

        node_input = input + nodevec_fusion
        node_input = self.time_norm(node_input)

        idx = torch.arange(self.dynamic_nodes).to(self.device)
        nodevec_static = self.embed1(idx)

        # Node fusion module
        nodevec_fusion = self.gate_Fusion_1(node_input) + nodevec_static

        # Graph learning module
        base_adj = self.graph_learn(nodevec_fusion)

        # 保存当前输出（作为新任务输出）
        self.new_model_output = base_adj

        return base_adj

    def lwf_update(self, input, incre):
        """
        LwF 增量学习更新方法
        """
        # 1. 前向计算
        current_output = self.forward(input, incre)

        # 2. 计算新任务损失
        new_task_loss = torch.mean(current_output)  # 示例任务损失

        # 3. 计算旧任务损失（保持与旧模型输出的一致性）
        if self.old_model_output is not None:
            old_task_loss = F.kl_div(
                F.log_softmax(current_output, dim=-1),
                F.softmax(self.old_model_output, dim=-1),
                reduction='batchmean'
            )
        else:
            old_task_loss = torch.tensor(0.0, device=self.device)

        # 4. 总损失
        total_loss = new_task_loss + self.lwf_lambda * old_task_loss

        # 5. 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 保存当前模型输出作为旧任务的参考
        self.old_model_output = current_output.detach()

        return total_loss.item()
