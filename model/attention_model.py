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
    def attention(query, key, value, mask):
        d_k = key.size(-1)    # feature size
        # print(query.size(),key.size())
        scores = torch.matmul(query, key.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(d_k))

        if mask is not None:
            # print(scores.shape,mask.shape)
            # todo 一会这里打个断点，看mask是不是有问题
            scores = scores.masked_fill(mask == 0, -1e20)

        attn = fn.softmax(scores, dim=-1)

        return torch.matmul(attn, value)

    def start(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

# todo: 这里考虑一下，感觉参数和节点的聚合器既然已经分开写了那就直接把size去掉就行了，维度就是(1,1,1,feature_size) (这个已经实现了，考虑结束后把size去掉，并且根据聚合器的不同给定了q的维度)


class ParameterAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        self._q = nn.Parameter(torch.randn(
                (1, 1, 1, args.hidden_dim)), requires_grad=True)
        self.start()

    @property
    def parameter(self):
        return self._q

    @parameter.setter
    def parameter(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f'parameter must be a torch.Tensor, but now is {type(value)}')
        if value.dim() != 4:
            raise ValueError(
                f'parameter must be a 4-dim tensor, but now is {value.dim()}')
        self._q = nn.Parameter(value,requires_grad=True)

    def forward(self, data, mask=None):
        # print(data.shape,mask.shape)
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
        query = self._q.repeat(batch_size,node_size,1,1)
        query = query.reshape(batch_size, node_size, -1,
                               self.n_head, self.head_dim).transpose(-2, -3)
        mask = mask.unsqueeze(-3)

        x = self.attention(query, key, value, mask)

        x = x.transpose(-2, -3).contiguous().reshape(batch_size,
                                                     node_size, -1, self.n_head*self.head_dim)
        x = torch.squeeze(x)

        return self.dense(x)


# todo: 这个后面还要写一下聚合node的版本，还有后续还要解决一下graph的聚合的维度问题 (好像已经完成了)
class NodeAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        if args.use_q_node:
            self._q = nn.Parameter(torch.randn(
                (args.batch_size, 1, args.hidden_dim)), requires_grad=True)  # 这里就算用自注意力为了保持一致还是应该用batch_size大小的矩阵
        else:
            self._q = nn.Parameter(torch.randn(
                (1, 1, args.hidden_dim)), requires_grad=False)


        if args.add_par_pos_emb:
            self.w_k = nn.Linear(args.hidden_dim*2, args.hidden_dim, bias=False)
            self.w_v = nn.Linear(args.hidden_dim*2, args.hidden_dim, bias=False)

        self.start()


    @property
    def parameter(self):
        return self._q

    @parameter.setter
    def parameter(self, value):
        
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f'parameter must be a torch.Tensor, but now is {type(value)}')
        if value.dim() != 3:
            raise ValueError(
                f'parameter must be a 3-dim tensor, but now is {value.dim()}')
        self._q = nn.Parameter(value,requires_grad=True)

    def forward(self, data, mask=None):
        batch_size = data.shape[0]
        feature_size = args.hidden_dim

        self.head_dim = feature_size//self.n_head
        if self.head_dim*self.n_head != feature_size:
            raise ValueError(
                f'feature_size must be divisible by n_head, but now feature_size is {feature_size}, n_head is {self.n_head}')

        key = self.w_k(data).reshape(batch_size, -1,
                                     self.n_head, self.head_dim).transpose(-2, -3)
        value = self.w_v(data).reshape(batch_size, -1,
                                       self.n_head, self.head_dim).transpose(-2, -3)
        if self._q.shape[0]==1:
            query = self._q.repeat(batch_size,1,1)
            query = query.reshape(batch_size, -1,
                               self.n_head, self.head_dim).transpose(-2, -3)
        else:
            query = self._q.reshape(batch_size, -1,
                               self.n_head, self.head_dim).transpose(-2, -3)
        mask = mask.unsqueeze(-3)

        x = self.attention(query, key, value, mask)

        x = x.transpose(-2, -3).contiguous().reshape(batch_size, -
                                                     1, self.n_head*self.head_dim)
        x = torch.squeeze(x)

        return self.dense(x)


class StructureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = nn.ModuleList([
            dglnn.GINConv(nn.Linear(args.hidden_dim+args.op_type_size, args.hidden_dim),
                          aggregator_type='sum', activation=nn.ReLU()),
            dglnn.GINConv(nn.Linear(args.hidden_dim, args.hidden_dim),
                          aggregator_type='sum', activation=nn.ReLU()),
        ])

    def forward(self, graph_data: dgl.DGLGraph):
        x = graph_data.ndata['struct_feature']
        for conv_gin in self.extractor:
            x = conv_gin(graph_data, x)
        
        graph_data.ndata['struct_feature']=x

        # 准备使用dgl给出的方法进行聚合来降维，下面的是使用self_attention的方式来降维
        # batch_num_nodes = graph_data.batch_num_nodes()
        # cumsum = torch.cumsum(torch.tensor(
        #     [0] + batch_num_nodes.tolist()), dim=0)

        # features_list = list()
        # for i in range(len(batch_num_nodes)):  # 拆分大图，用于后面的特征降维
        #     start_idx = cumsum[i]
        #     end_idx = cumsum[i + 1]
        #     subgraph_features = graph_data.ndata['feature'][start_idx:end_idx]
        #     features_list.append(subgraph_features)

        # features = torch.FloatTensor(features_list)
        return graph_data

class GraphStructureAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor=StructureExtractor()
        self.dense=nn.Linear(args.hidden_dim*3,args.hidden_dim)

    def forward(self,graph_data):
        graph_data=self.extractor(graph_data)

        mean_feature=dgl.mean_nodes(graph_data,'struct_feature')
        max_feature=dgl.max_nodes(graph_data,'struct_feature')
        sum_feature=dgl.sum_nodes(graph_data,'struct_feature')

        feature=torch.concat([mean_feature,max_feature,sum_feature],dim=-1)

        structure_feature=self.dense(feature)

        return structure_feature

class StructureAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(
            (1, 1, args.hidden_dim)), requires_grad=True)
        if torch.cuda.is_available():
                self.q=self.q.to(torch.device('cuda'))
        self.start()

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
        mask = mask.unsqueeze(-3)

        x = self.attention(query, key, value, mask)

        x = x.transpose(-2, -3).contiguous().reshape(batch_size, -
                                                     1, self.n_head*self.head_dim)
        x = torch.squeeze(x)

        return self.dense(x)


# todo:后面还要确定一下图节点的参数的维度，在gin的输入维度处填写，这里先随便填了一个不存在的args的参数
# todo:这里还有一个问题就是我把图神经网络只放了两层，后续如果要调这个超参的话，可以加一个args的参数
class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_aggregator = NodeAggregator()
        self.parameter_aggregator = ParameterAggregator()
        self.structure_aggregator = GraphStructureAggregator()


        self.par_dense = nn.Linear(args.bin_num-1, args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2),
        )

    def forward(self, dgl_graph, par_tensor, row_mask=None, node_mask=None):
        par_feature = self.par_dense(par_tensor)  # 改变参数维度到hidden_dim
        archi_feature = self.structure_aggregator(dgl_graph)   # 改变结构参数维度到hidden_dim
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  # 扩充第二维用于作为query
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask)  # 聚合row_size
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature   # 设置query
        
        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  # 聚合node_size

        # print(node_feature.shape)
        res=self.classifier(node_feature)
        # print(res.shape)
        return res


class ModelTestNodeAggregate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter_aggregator = ParameterAggregator()
        self.node_aggregator = NodeAggregator()

        self.par_dense = nn.Linear(args.bin_num-1, args.hidden_dim)
        self.archi_dense = nn.Linear(3, args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2),
        )

    def forward(self, data, archi_feature, row_mask=None, node_mask=None):
        par_feature = self.par_dense(data)  # 改变参数维度到hidden_dim
        archi_feature = self.archi_dense(archi_feature)   # 改变结构参数维度到hidden_dim
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  # 扩充第二维用于作为query
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask)  # 聚合row_size
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature   # 设置query
        
        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  # 聚合node_size

        # print(node_feature.shape)
        res=self.classifier(node_feature)
        # print(res.shape)
        return res
    

class ModelTestNodePosEmbAggregate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter_aggregator = ParameterAggregator()
        self.node_aggregator = NodeAggregator()

        self.par_dense = nn.Linear(args.bin_num-1, args.hidden_dim)
        self.archi_dense = nn.Linear(3, args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2),
        )

    def forward(self, data, archi_feature, row_mask=None, node_mask=None, pos_emb=None):
        par_feature = self.par_dense(data)  # 改变参数维度到hidden_dim
        archi_feature = self.archi_dense(archi_feature)   # 改变结构参数维度到hidden_dim
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  # 扩充第二维用于作为query
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask)  # 聚合row_size
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature   # 设置query
        
        par_feature=torch.concat((pos_emb,par_feature),dim=-1)


        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  # 聚合node_size

        # print(node_feature.shape)
        res=self.classifier(node_feature)
        # print(res.shape)
        return res
