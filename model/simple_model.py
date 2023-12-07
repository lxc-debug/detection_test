import torch
import torch.nn as nn
from config.conf import args


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        k = self.k_trans(input_tensor)
        v = self.v_trans(input_tensor)

        attention = torch.matmul(self.q, torch.transpose(k, 1, 2))
        attention = torch.nn.functional.softmax(attention, dim=-1)
        res = torch.matmul(attention, v)
        res = torch.squeeze(res)
        res = self.classifier(res)
        return res
    
    def start(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias,0)
