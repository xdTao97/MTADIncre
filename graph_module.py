import torch
import torch.nn.functional as F
from networkx.classes import neighbors

from layer_module import *
import math
from sklearn.decomposition import PCA

class AdaptiveGraphLearning(nn.Module):

    def __init__(self, in_dim, heads, out_dim, n_neighbors, nodes, dropout=0.5):
        super(AdaptiveGraphLearning, self).__init__()
        self.D = heads * out_dim
        self.heads = heads
        self.dropout = dropout
        self.in_dim = in_dim
        self.nodes = nodes
        self.out_dim = out_dim
        self.n_neighbors = n_neighbors  # 可变的邻居数
        self.attn_static = nn.LayerNorm(nodes)
        # 定义线性变换层
        self.query = nn.Linear(self.in_dim, self.D)
        self.key = nn.Linear(self.in_dim, self.D)
        self.value = nn.Linear(self.in_dim, self.D)
        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1))

        self.attn_norm = nn.LayerNorm(nodes)

        self.soft_threshold = SoftThreshold(0)

        # 动态邻接矩阵的权重
        self.attn_linear = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear.data)

    def static_graph(self, nodevec):
        # print("nodevec shape:", nodevec.shape)
        resolution_static = torch.bmm(nodevec, nodevec.transpose(2, 1))
        resolution_static = F.softmax(F.relu(self.attn_static(resolution_static)), dim=1)
        return resolution_static

    def forward(self, nodevec_fusion):
        alpha = 0.4
        batch_size = nodevec_fusion.size(0)
        nodes = self.nodes
        # Static Graph Structure Learning
        #adj_static = self.static_graph(nodevec_fusion)
        adj_static = self.static_graph(nodevec_fusion)

        # Step 1: 特征计算
        query = self.query(nodevec_fusion).squeeze(-1).transpose(1, -1)  # (batch_size, 1, heads, head_dim, nodes)
        key = self.key(nodevec_fusion).squeeze(-1).transpose(1, -1)

        key = key.view(batch_size, self.heads, self.out_dim, nodes)
        query = query.view(batch_size, self.heads, self.out_dim, nodes).transpose(-1, -2)

        # Step 2: 计算节点之间的注意力
        '''
        attention = torch.einsum('bhnd, bhdu->bhnu', query, key)  # (batch_size, heads, nodes, nodes)
        attention = attention / (self.out_dim ** 0.5)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention = self.mlp(attention) + attention
        '''
        chunk_size = math.ceil(nodes / 2)  # 这里是把节点拆分成多个块；拆分的块要和特征成倍数关系
        q_chi = torch.split(query, chunk_size, dim=-1)  # 拆分 query
        k_chi = torch.split(key, chunk_size, dim=-1)  # 拆分 key

        output = []
        for q, k in zip(q_chi, k_chi):
            attention = torch.einsum('bhnd, bhdu->bhnu', q, k)  # (batch_size, heads,nodes, nodes)
            attention = attention / (self.out_dim ** 0.5)
            attention = F.dropout(attention, self.dropout, training=self.training)
            attention = self.soft_threshold(attention)

            attention = F.relu(self.mlp(attention)) + attention
            output.append(attention)

        attention = torch.cat(output, dim=-1)  # 拼接成完整的注意力矩阵

        # attention = output
        # Step 3: 计算每对节点之间的相似度
        attention_scores = attention.sum(dim=1)  # (batch_size, nodes, nodes)


        # Step 4: 动态选择最相关的n个特征
        # 计算每个节点之间的相似度（这里的注意力得分可以视作特征之间的相似度）
        topk_values, topk_indices = torch.topk(attention_scores, self.n_neighbors, dim=-1, largest=True, sorted=True)

        # Step 5: 构建新的邻接矩阵
        adj_dynamic = torch.zeros(batch_size, nodes, nodes).to(nodevec_fusion.device)

        combined_adj = adj_dynamic.scatter_add_(-1, topk_indices, topk_values)  # 将选择的top-k值赋给邻接矩阵

        # combined_adj =  alpha * new_adj_dynamic + (1-alpha) * adj_static

        return combined_adj

class SoftThreshold(nn.Module):
    def __init__(self, beta):
        super(SoftThreshold, self).__init__()
        self.beta = beta

    def forward(self, input):
        return torch.sign(input) * torch.relu(torch.abs(input) - self.beta)

class IncrementalGraphConstructor(nn.Module):
    def __init__(self, nodes, dim, out_dim, device, time_step, neighbors, dynamic_nodes, cout=16, heads=4,
                  dropout=0.5, m=0.9, in_dim=256, is_add1=True):

        super(IncrementalGraphConstructor, self).__init__()
        self.embed1 = nn.Embedding(nodes, dim)
        self.m = m
        self.embed2 = nn.Embedding(nodes, dim)
        self.neighbors = neighbors
        for param in self.embed2.parameters():
            param.requires_grad = False
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_w.data

        self.device = device
        self.nodes = nodes
        self.time_step = time_step
        self.dynamic_nodes = dynamic_nodes  # For dynamic node management

        if is_add1:
            time_length = time_step + 1
        else:
            time_length = time_step

        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, 25), bias=True)  # cout
        self.gate_Fusion_1 = gatedFusion_1(dim, device)


        self.graph_learn = AdaptiveGraphLearning(in_dim=dim, heads=heads, out_dim=out_dim,
                                                 n_neighbors=neighbors, nodes=nodes, dropout=dropout)

        self.dim_to_channels = nn.Parameter(torch.zeros(size=(heads * out_dim, cout * time_step)))
        nn.init.xavier_uniform_(self.dim_to_channels.data, gain=1.414)
        self.skip_norm = nn.LayerNorm(time_step)
        self.time_norm = nn.LayerNorm(normalized_shape=[dim])
        self.time_norm1 = nn.LayerNorm(normalized_shape=[24])
        self.time_norm2 = nn.LayerNorm(normalized_shape=[1])
        self.pca_model = PCA(self.neighbors)
        self.is_fitted = False  # 用于标识 PCA 是否已经拟合
        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, 25), bias=True)  # cout

    def fit_graph(self, adj_matrices):
        """
        对邻接矩阵进行 PCA 拟合。
        Args:
            adj_matrices (Tensor): 输入邻接矩阵，形状为 (batch_size, nodes, nodes)。
        """

        batch_size, nodes, _ = adj_matrices.shape
        adj_list = []

        # 将邻接矩阵转换为 NumPy 格式
        for i in range(batch_size):
            adj = adj_matrices[i].cpu().numpy()
            adj_list.append(adj)

        adj_concat = torch.cat([torch.tensor(a) for a in adj_list]).cpu().numpy()
        self.pca_model.fit(adj_concat)  # 对拼接后的数据进行拟合
        self.is_fitted = True

    def condense_graph(self, combined_adj):
        # print(f"Input shape: {combined_adj.shape}")

        batch_size, nodes, _ = combined_adj.shape
        compressed_adjs = []

        if not self.is_fitted:
            # 自动拟合 PCA 模型
            # print("PCA model not fitted. Automatically calling 'fit_graph()'.")
            self.fit_graph(combined_adj)

        for i in range(batch_size):
            adj = combined_adj[i].detach().cpu().numpy()             # 将 Tensor 转为 NumPy 数组
            reduced_adj = self.pca_model.transform(adj)     # 使用已经拟合的 PCA 模型进行变换
            compressed_adjs.append(torch.tensor(reduced_adj, dtype=torch.float).to(self.device))

        return torch.stack(compressed_adjs)  # 堆叠成形状为 (batch_size, nodes, n_components)

    def forward(self, input ,incre):

        batch_size, nodes, time_step = input.shape[0], self.dynamic_nodes, self.time_step
        # Momentum update for embeddings
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        base_adj_incre = None
        if incre:
            # Split datasets into raw and incremental datasets
            part1, part2 = torch.split(input, [24, 1], dim=2)

            input = torch.cat([part1, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part1.shape[2],
                                                     device=self.device)], dim=2)
            part2_incre = torch.cat([part2, torch.zeros(input.shape[0], input.shape[1], input.shape[2] - part2.shape[2],
                                                        device=self.device)], dim=2)

            # **增量处理**：动态生成额外特征并修正图学习结果
            node_input_incre = part2_incre
            nodevec_fusion = self.time_norm(node_input_incre)


            # Node fusion module
            #nodevec_fusion = self.gate_Fusion_1(batch_size, nodevec_static, node_input_incre) + nodevec_static
        else:
            nodevec_fusion = torch.zeros(input.shape[0],input.shape[1],input.shape[2], device=self.device)

        node_input = input + nodevec_fusion

        node_input = self.time_norm(node_input)

        idx = torch.arange(self.dynamic_nodes).to(self.device)
        nodevec_static = self.embed1(idx)

        # Node fusion module
        nodevec_fusion = self.gate_Fusion_1(batch_size, nodevec_static, node_input) + nodevec_static

        # Graph learning module (dynamic adjacency matrix)
        base_adj = self.graph_learn(nodevec_fusion)

        return base_adj


