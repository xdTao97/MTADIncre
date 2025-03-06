import torch

from incre.gem_graph import *
from graph_module import *
from incre.ERGnn_graph import *
from incre.SSMgraph import *
from layer_mtgnn import *
from layer_module import *
from MSLIncre import *

'''
# 增加图压缩时的操作
class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        x = x.unsqueeze(0)
        # 查看节点特征的形状
        x=x.repeat(A.shape[0],1, 1, 1).sum(dim=1)  # 将 x 扩展为 [256, 90, 25]

        #x = torch.einsum('nvw, nw->nvl', A, x)
        return x.contiguous()
        # graph_data = torch.tensor(graph_data)
        # return graph_data
    
'''


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


#不加图压缩时的操作
class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        x = x.unsqueeze(0)
        # 查看节点特征的形状

        x = x.repeat(A.shape[0], 1, 1, 1).sum(dim=1)  # 将 x 扩展为 [256, 90, 25]

        x = torch.einsum('nvw, nwl->nvl', A, x)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_module(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_module, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = []

        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(1, self.order):
            x2 = self.nconv(x1, support)
            out.append(x2)
            # x1 = x2

        h = F.dropout(x1, self.dropout, training=self.training)
        return h


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = dnconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class MTAD_model(nn.Module):
    def __init__(self, device, num_nodes, dropout, in_dim, out_dim, n_neighbors, batch_size, dynamic_nodes, gcn_bool,
                 addaptadj=True,
                 residual_channels=32, dilation_channels=32, skip_channels=64,
                 end_channels=256, layers=2, dropout_ingc=0.5, eta=1, gamma=0.001, seq_length=12,
                 m=0.9, dilation_exponential_=1):
        super(MTAD_model, self).__init__()
        self.dropout = dropout
        self.n_neighbors = n_neighbors
        self.in_dim = in_dim  #features
        self.seq_length = seq_length
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.batch_size = batch_size
        self.device = device

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv_s = nn.ModuleList()
        self.gconv_d = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=batch_size,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        #self.seq_length = seq_length
        kernel_size = 7

        dilation_exponential = dilation_exponential_
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        rf_size_i = 1
        new_dilation = 1
        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                # rf_size_j = 7, 19, 43, 91, 187
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.filter_convs.append(
                dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
            self.gate_convs.append(
                dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
            self.deepConve = DepthWiseConv(num_nodes, in_dim)
            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            if self.seq_length > self.receptive_field:

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

            if self.gcn_bool:
                self.gconv_s.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))
                self.gconv_d.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))

            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm((batch_size, num_nodes, in_dim), eps=1e-5,
                                           elementwise_affine=True))
            else:
                self.norm.append(LayerNorm((batch_size, num_nodes, in_dim), eps=1e-5,
                                           elementwise_affine=True))
            new_dilation *= dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=batch_size,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.batch_size, out_channels=skip_channels, kernel_size=(1, in_dim),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)

        else:

            self.skip0 = nn.Conv2d(in_channels=self.batch_size, out_channels=skip_channels,
                                   kernel_size=(1, out_dim), bias=True)
            self.skipE = nn.Conv2d(in_channels=batch_size, out_channels=skip_channels,
                                   kernel_size=(1, out_dim), bias=True)

        self.idx = torch.arange(self.nodes).to(device)

        #self.graph_construct = graph_constructor(num_nodes, in_dim, device, batch_size, n_neighbors, cout=16,
         #        heads=4, head_dim=8, dropout=dropout_ingc, m=m, in_dim=256, is_add1=True)

        self.graph_construct = IncrementalGraphConstructor(num_nodes, in_dim, out_dim, device, batch_size, n_neighbors,
              dynamic_nodes, cout=16, heads=4, dropout=dropout_ingc, m=m, is_add1=True)
        # self.graph_construct = MSLIncre(num_nodes, in_dim, out_dim, device, batch_size, dynamic_nodes, m=m, heads=4)
        # self.graph_construct = IncrementalModelWithGEM(num_nodes, in_dim, out_dim, device, batch_size,dynamic_nodes,
         #                                    cout=16, heads=4, dropout=dropout_ingc, m=m, is_add1=True)
        #self.graph_construct = IncrementalModelWithERGNN(num_nodes, in_dim, out_dim, device, batch_size, dynamic_nodes)
        # self.graph_construct = IncreSSM(num_nodes, in_dim, out_dim, device, batch_size,dynamic_nodes, m=m, heads=4)
        # self.pool = nn.AdaptiveAvgPool2d((out_dim, 1))
        # self.pool = nn.AdaptiveAvgPool1d(out_dim)

        self.fc = nn.Linear(num_nodes + self.in_dim, out_dim)
        # self.fc = nn.Linear(num_nodes, out_dim)

        self.temporal_embedding = nn.Parameter(torch.randn(1, 2))

    def forward(self, input):
        #input shape (b, n, k): b - batch size, n - window size, k - number of features
        #static_adj,  adj_d = None, None
        compressed_graph = None
        # 空间维度
        if self.gcn_bool:
            compressed_graph = self.graph_construct(input, True)

        # 时间维度
        # 如果最后一个批次不足目标批次大小，则填充
        input = data_pad(input, self.batch_size)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        x = self.start_conv(input)
        residual = x

        #x = self.temporal_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # [batch_size, 1, d_model]
        # WaveNet layers
        for i in range(1, self.layers):
            # dilated convolution
            filter = self.deepConve(x)
            filter = self.filter_convs[i](x)
            filter = torch.relu(filter)
            # gate = self.deepConve(x)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_bool:
                x_d = self.gconv_d[i](x, compressed_graph)
                x = x_d
            else:
                x = self.residual_convs[i](x)
            residual_expanded = residual.repeat(x.size(0) // residual.size(0), 1, 1)  # 将 batch size 扩展为 256
            residual_expanded = data_pad(residual_expanded, self.batch_size)
            x = data_pad(x, self.batch_size)
            x = x + residual_expanded[:, :, -x.size(1):]
            x = self.norm[i](x, None)

        adj_d = data_pad(compressed_graph, self.batch_size)

        x2 = torch.cat((adj_d,x),2)
        x2 = F.relu(x2)
        x3 = x2.mean(dim=1)
        predict = self.fc(x3)  # 输出维度为[batch_size, output_dim]
        return predict


def data_pad(input, batch_size):
    padding_size = batch_size - input.size(0)
    if padding_size > 0:
        padding = (0, 0, 0, 0, 0, padding_size)  # 填充批次维度
        input = F.pad(input, padding, "constant", 0)
    return input
