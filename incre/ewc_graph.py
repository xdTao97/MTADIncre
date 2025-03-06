import torch
import torch.nn as nn
import torch.optim as optim
from layer_module import *

class IncrementalModelWithEWC(nn.Module):
    # def __init__(self, dynamic_nodes, time_step, device, loss_fn=nn.MSELoss(), ewc_lambda=0.4):
    def __init__(self, num_nodes, in_dim, out_dim, device, time_step, dynamic_nodes,loss_fn=nn.MSELoss(), ewc_lambda=0.4):
        super(IncrementalModelWithEWC, self).__init__()
        self.dynamic_nodes = dynamic_nodes
        self.time_step = time_step
        self.device = device
        self.loss_fn = loss_fn
        self.ewc_lambda = ewc_lambda
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        # 时间归一化模块
        self.time_norm = nn.LayerNorm(normalized_shape=[in_dim])

        # Embedding layers
        self.embed1 = nn.Embedding(dynamic_nodes, in_dim)
        self.embed2 = nn.Embedding(dynamic_nodes, in_dim)
        self.m = 0.9  # Momentum coefficient

        # Node fusion module (example)
        self.gate_Fusion_1 = gatedFusion_1(in_dim, device)
        self.feat_expand = nn.Linear(self.in_dim, self.num_nodes)
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        # EWC-specific variables
        self.fisher_information = {}
        self.prev_params = {}
        self.initialize_ewc()

    def initialize_ewc(self):
        """
        初始化 EWC 相关变量
        """
        for name, param in self.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param).to(self.device)
            self.prev_params[name] = param.detach().clone()

    def update_fisher_and_params(self, dataloader):
        """
        更新 Fisher 信息矩阵和保存的参数
        """
        # 初始化 Fisher 信息矩阵
        for name, param in self.named_parameters():
            self.fisher_information[name].zero_()

        # 计算 Fisher 信息矩阵
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs, incre=False)
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.pow(2)

        # 平均化 Fisher 信息
        for name in self.fisher_information:
            self.fisher_information[name] /= len(dataloader)

        # 保存当前任务的参数
        for name, param in self.named_parameters():
            self.prev_params[name] = param.detach().clone()

    def ewc_update(self, nodevec_fusion):
        """
        EWC 更新方法
        """

        # 扩展输入特征维度
        nodevec_fusion_expanded = self.feat_expand(nodevec_fusion)  # (256, 100, 100)
        # 计算当前任务的损失
        current_loss = torch.mean(nodevec_fusion_expanded)  # 示例任务损失

        # 计算 EWC 正则项
        ewc_loss = 0.0
        for name, param in self.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")
            if name in self.fisher_information:
                fisher = self.fisher_information[name].to(self.device)
                prev_param = self.prev_params[name].to(self.device)
                ewc_loss += (fisher * (param - prev_param).pow(2)).sum()

        # 总损失 = 当前任务损失 + EWC 正则项
        total_loss = current_loss + self.ewc_lambda * ewc_loss

        # 使用总损失更新模型
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return nodevec_fusion

    def forward(self, input, incre):
        batch_size, nodes, time_step = input.shape[0], self.dynamic_nodes, self.time_step

        # 动量更新嵌入
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        if incre:
            # 分割数据为原始部分和增量部分
            part1, part2 = torch.split(input, [1, 24], dim=2)
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
        # EWC update
        base_adj = self.ewc_update(nodevec_fusion)

        return base_adj
