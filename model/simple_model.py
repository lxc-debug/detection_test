import torch
import torch.nn as nn
from config.conf import args


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        """_summary_
            这里根据use_three会决定特征的维度，使用use_three就是认为输入数据的后面三维是结构的特征，比如(0,0,1);(0,1,0);(1,0,0)。不使用use_three就是不在数据中引入结构信息
        """
        super().__init__()
        if args.use_three:
            self.k_trans = nn.Linear(args.bin_num+2, args.hidden_dim)
            self.v_trans = nn.Linear(args.bin_num+2, args.hidden_dim)
        else:
            self.k_trans = nn.Linear(args.bin_num-1, args.hidden_dim)
            self.v_trans = nn.Linear(args.bin_num-1, args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2)
        )

        self.q = nn.init.normal_(nn.Parameter(torch.randn(
            1, 1, args.hidden_dim), requires_grad=True))

        self.start()

    def __call__(self, input_tensor):
        """_summary_
            这个就是直接对row_size进行降维(使用self_attention的方式)，然后使用MLP进行分类
        Arguments:
            input_tensor -- 输入的tensor

        Returns:
            输出的结果
        """
        d_k = input_tensor.shape[-1]
        k = self.k_trans(input_tensor)
        v = self.v_trans(input_tensor)

        attention = torch.matmul(self.q, torch.transpose(
            k, 1, 2))/torch.sqrt(torch.tensor(d_k))
        attention = torch.nn.functional.softmax(attention, dim=-1)
        res = torch.matmul(attention, v)
        res = torch.squeeze(res)
        res = self.classifier(res)
        return res

    def start(self):
        """_summary_
            初始化模型的参数
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


class SimpleModelQ(nn.Module):
    def __init__(self) -> None:
        """_summary_
            这个是针对使用模型结构来作为query的情况的验证模型
        """
        super().__init__()
        self.k_trans = nn.Linear(args.bin_num-1, args.hidden_dim)
        self.v_trans = nn.Linear(args.bin_num-1, args.hidden_dim)
        self.q_trans = nn.Linear(3, args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//2, args.hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.hidden_dim//4, 2)
        )

        self.start()

    def __call__(self, input_tensor):
        """_summary_
            首先输入的后三维还是模型的结构特征，进行提取后使用一个线性变换到hidden_dim的维度。然后使用结构特征作为query进行降维，之后使用MLP进行分类
        Arguments:
            input_tensor -- 模型的输入

        Returns:
            输入的结果
        """
        d_k = input_tensor.shape[-1]
        input_tensor, archi_feature_arr = torch.split(
            input_tensor, args.bin_num-1, dim=-1)
        archi_feature = torch.unsqueeze(archi_feature_arr[:, 0, :], dim=1)

        k = self.k_trans(input_tensor)
        v = self.v_trans(input_tensor)
        q = self.q_trans(archi_feature)

        attention = torch.matmul(q, torch.transpose(
            k, 1, 2))/torch.sqrt(torch.tensor(d_k))
        attention = torch.nn.functional.softmax(attention, dim=-1)
        res = torch.matmul(attention, v)
        res = torch.squeeze(res)
        res = self.classifier(res)
        return res

    def start(self):
        """_summary_
            模型参数的初始化
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
