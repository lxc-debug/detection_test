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
            scores = scores.masked_fill(mask == 0, -1e20)

        attn = fn.softmax(scores, dim=-1)

        return torch.matmul(attn, value)

    def start(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)




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


class NodeAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
        if args.use_q_node:
            self._q = nn.Parameter(torch.randn(
                (args.batch_size, 1, args.hidden_dim)), requires_grad=True)  
        else:
            self._q = nn.Parameter(torch.randn(
                (1, 1, args.hidden_dim)), requires_grad=False)


        if args.add_par_pos_emb:
            self.w_k = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
            self.w_v = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)

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



class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_aggregator = NodeAggregator()
        self.parameter_aggregator = ParameterAggregator()
        self.structure_aggregator = GraphStructureAggregator()


        self.par_dense = nn.Linear(args.bin_num-1, args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim*2, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2),
        )

    def forward(self, dgl_graph, par_tensor, row_mask=None, node_mask=None):
        par_feature = self.par_dense(par_tensor)  
        archi_feature = self.structure_aggregator(dgl_graph)   
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask)  
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature   
        
        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  

        feature=torch.concat([node_feature,torch.squeeze(archi_feature)],dim=-1)

        # print(node_feature.shape)
        res=self.classifier(feature)
        # print(res.shape)
        return res

class PosMainModel(nn.Module):
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

    def forward(self, dgl_graph, par_tensor, row_mask=None, node_mask=None, pos_emb=None):
        par_feature = self.par_dense(par_tensor)  
        archi_feature = self.structure_aggregator(dgl_graph)   
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask)  
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature   
        
        # par_feature=torch.concat((pos_emb,par_feature),dim=-1)
        par_feature=pos_emb+par_feature

        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  

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
        self.archi_dense = nn.Linear(9, args.hidden_dim)

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
        par_feature = self.par_dense(data)  
        archi_feature = self.archi_dense(archi_feature)  
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask)  
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature  
        
        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  

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
        par_feature = self.par_dense(data)  
        archi_feature = self.archi_dense(archi_feature)   
        archi_feature = torch.unsqueeze(archi_feature, dim=1)  
        # print(archi_feature.shape)

        par_feature = self.parameter_aggregator(
            par_feature, mask=row_mask) 
        
        # print(par_feature.shape)
        if args.use_q_node:
            self.node_aggregator.parameter=archi_feature 
        
        par_feature=torch.concat((pos_emb,par_feature),dim=-1)


        node_feature = self.node_aggregator(
            par_feature, mask=node_mask)  

        # print(node_feature.shape)
        res=self.classifier(node_feature)
        # print(res.shape)
        return res
