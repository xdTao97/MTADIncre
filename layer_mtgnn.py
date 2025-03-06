from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(1):]
        # 确保所有张量的宽度一致
        #max_width = max([tensor.size(3) for tensor in x])
        # 在拼接前打印所有张量的尺寸
        '''
        for idx, tensor in enumerate(x):
            print(f"x[{idx}] size: {tensor.size()}")
        '''

        # 计算最大宽度
        max_width = max([tensor.size(2) for tensor in x])+1
        #print(f"max_width: {max_width}")

        # 对每个张量进行填充，确保宽度一致
        for idx, tensor in enumerate(x):
            if tensor.size(2) < max_width:
                pad_size = max_width - tensor.size(2)
                x[idx] = F.pad(tensor, (0, pad_size))  # 在宽度上填充



        x = torch.cat(x, dim=0)

        return x


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # (residual_channels, num_nodes, self.seq_length - rf_size_j + 1) 这是第一个参数
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        inputrer = input.shape
        weight = self.weight[:, :]
        bias = self.bias[:, :]
        if self.elementwise_affine:
            return F.layer_norm(input, inputrer, weight, bias, self.eps)

        else:
            return F.layer_norm(input, inputrer,  weight, bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
