import torch
from torch import Tensor
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from config.conf import args
import dgl
import torch.nn.functional as fn
from abc import ABC, abstractmethod


class Aggregator(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.w_k = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.w_v = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.dense = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.n_head = args.n_head

    @abstractmethod
    def forward(self, data, mask=None):
        pass

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        d_k = key.size(-1)    # feature size
        scores = torch.matmul(query, key.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(d_k))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)

        attn = fn.softmax(scores, dim=-1)

        return torch.matmul(attn, value)

# todo: 这里考虑一下，感觉参数和节点的聚合器既然已经分开写了那就直接把size去掉就行了，维度就是(1,1,1,feature_size) (这个已经实现了，考虑结束后把size去掉，并且根据聚合器的不同给定了q的维度)


class ParameterAggregator(Aggregator):
    def __init__(self, parameter=None) -> None:
        super().__init__()
        self.q = parameter

    @property
    def q(self):
        return self.q

    @q.setter()
    def set_q(self, parameter):
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(
                f'parameter must be a torch.Tensor, but now is {type(parameter)}')
        if parameter.dim() != 4:
            raise ValueError(
                f'parameter must be a 4-dim tensor, but now is {parameter.dim()}')
        self.q = parameter

    def forward(self, data, mask=None):
        if not self.q:
            self.q = nn.Parameter(torch.randn(
                (1, 1, 1, args.hidden_dim)), requires_grad=True)

        batch_size = data.shape[0]
        node_size = data.shape[1]
        feature_size = data.shape[-1]
        self.head_dim = feature_size//self.n_head
        if self.head_dim*self.n_head != feature_size:
            raise ValueError(
                f'feature_size must be divisible by n_head, but now feature_size is {feature_size}, n_head is {self.n_head}')

        key = self.w_k(data).reshape(batch_size, node_size, -1,
                                     self.n_head, self.head_dim).transpose(-2, -3)
        value = self.w_v(data).reshape(batch_size, node_size, -1,
                                       self.n_head, self.head_dim).transpose(-2, -3)
        query = self.q.reshape(batch_size, node_size, -1,
                               self.n_head, self.head_dim).transpose(-2, -3)
        mask = mask.reshape(batch_size, node_size, -1,
                            self.n_head, self.head_dim).transpose(-2, -3)

        x = self.attention(query, key, value, mask)

        x = x.transpose(-2, -3).contiguous().reshape(batch_size,
                                                     node_size, -1, self.n_head*self.head_dim)
        x = torch.squeeze(x)

        return self.dense(x)


# todo: 这个后面还要写一下聚合node的版本，还有后续还要解决一下graph的聚合的维度问题 (好像已经完成了)
class NodeAggregator(Aggregator):
    def __init__(self, parameter=None) -> None:
        super().__init__()
        self.q = parameter

    @property
    def q(self):
        return self.q

    @q.setter()
    def set_q(self, parameter):
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(
                f'parameter must be a torch.Tensor, but now is {type(parameter)}')
        if parameter.dim() != 3:
            raise ValueError(
                f'parameter must be a 3-dim tensor, but now is {parameter.dim()}')
        self.q = parameter

    def forward(self, data, mask=None):
        if not self.q:
            self.q = nn.Parameter(torch.randn(
                (1, 1, args.hidden_dim)), requires_grad=True)  # 这就说明要使用自注意力机制来降维了
        batch_size = data.shape[0]
        feature_size = data.shape[-1]
        self.head_dim = feature_size//self.n_head
        if self.head_dim*self.n_head != feature_size:
            raise ValueError(
                f'feature_size must be divisible by n_head, but now feature_size is {feature_size}, n_head is {self.n_head}')

        key = self.w_k(data).reshape(batch_size, -1,
                                     self.n_head, self.head_dim).transpose(-2, -3)
        value = self.w_v(data).reshape(batch_size, -1,
                                       self.n_head, self.head_dim).transpose(-2, -3)
        query = self.q.reshape(batch_size, -1,
                               self.n_head, self.head_dim).transpose(-2, -3)
        mask = mask.reshape(batch_size, -1,
                            self.n_head, self.head_dim).transpose(-2, -3)

        x = self.attention(query, key, value, mask)

        x = x.transpose(-2, -3).contiguous().reshape(batch_size, -
                                                     1, self.n_head*self.head_dim)
        x = torch.squeeze(x)

        return self.dense(x)


class StructureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = nn.ModuleList([
            dglnn.GINConv(nn.Linear(args.graph_feature_dim, args.hidden_dim),
                          aggregator_type='sum', activation=nn.ReLU()),
            dglnn.GINConv(nn.Linear(args.hidden_dim, args.hidden_dim),
                          aggregator_type='sum', activation=nn.ReLU()),
        ])

    def forward(self, graph_data: dgl.DGLGraph, x):
        x = graph_data.ndata['feature']
        for conv_gin in self.extractor:
            x = conv_gin(graph_data, x)

        batch_num_nodes = graph_data.batch_num_nodes()
        cumsum = torch.cumsum(torch.tensor(
            [0] + batch_num_nodes.tolist()), dim=0)

        features_list = list()
        for i in range(len(batch_num_nodes)):  # 拆分大图，用于后面的特征降维
            start_idx = cumsum[i]
            end_idx = cumsum[i + 1]
            subgraph_features = graph_data.ndata['feature'][start_idx:end_idx]
            features_list.append(subgraph_features)

        features = torch.FloatTensor(features_list)
        return features


class StructureAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(
            (1, 1, args.hidden_dim)), requires_grad=True)

    def forward(self, data, mask=None):
        batch_size = data.shape[0]
        feature_size = data.shape[-1]
        self.head_dim = feature_size//self.n_head
        if self.head_dim*self.n_head != feature_size:
            raise ValueError(
                f'feature_size must be divisible by n_head, but now feature_size is {feature_size}, n_head is {self.n_head}')

        key = self.w_k(data).reshape(batch_size, -1,
                                     self.n_head, self.head_dim).transpose(-2, -3)
        value = self.w_v(data).reshape(batch_size, -1,
                                       self.n_head, self.head_dim).transpose(-2, -3)
        query = self.q.reshape(batch_size, -1,
                               self.n_head, self.head_dim).transpose(-2, -3)
        mask = mask.reshape(batch_size, -1,
                            self.n_head, self.head_dim).transpose(-2, -3)

        x = self.attention(query, key, value, mask)

        x = x.transpose(-2, -3).contiguous().reshape(batch_size, -
                                                     1, self.n_head*self.head_dim)
        x = torch.squeeze(x)

        return self.dense(x)


# todo:后面还要确定一下图节点的参数的维度，在gin的输入维度处填写，这里先随便填了一个不存在的args的参数
# todo:这里还有一个问题就是我把图神经网络只放了两层，后续如果要调这个超参的话，可以加一个args的参数
class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_aggregator = NodeAggregator()
        self.parameter_aggregator = ParameterAggregator()

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2),
        )

    def forward(dgl_graph, par_tensor, node_size, row_size):
        pass
