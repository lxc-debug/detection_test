import torch
import onnx
import numpy as np
from config.conf import args
import os


def model2onnx(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    save_dir: str = args.onnx_model,
) -> None:
    if model is None:
        model = torch.load(model_name)
        model = model.cpu()
    else:
        stat = torch.load(model_name)
        model.load_state_dict(stat)

    # 这里要判断一下模型的架构是否是对应的
    if args.use_archi:  
        if args.architecture != model_name.split('/')[-1].split('_')[1]:
            return
        onnx_model_path = os.path.join(
            save_dir, args.architecture, model_name.split('/')[-1].split('.')[0]+'.onnx')
    else:
        onnx_model_path = os.path.join(
            save_dir, model_name.split('/')[-1].split('.')[0]+'.onnx')

    if not os.path.exists(os.path.dirname(onnx_model_path)):
        os.makedirs(os.path.dirname(onnx_model_path))

    torch.onnx.export(model, input_tensor, onnx_model_path)


def onnx2dgl(
    model_path: str,
    is_poisoned: int = 1,
) -> tuple:
    # if args.use_archi:
    #     onnx_model_path = os.path.join(
    #         save_dir, args.architecture, model_name.split('/')[-1].split('.')[-2]+'.onnx')
    # else:
    #     onnx_model_path = os.path.join(
    #         save_dir, model_name.split('/')[-1].split('.')[-2]+'.onnx')

    onnx_model = onnx.load(model_path)

    graph = onnx_model.graph

    parameter_dict = dict()  # 网络参数

    for par in graph.initializer:  # 转换所有的网络参数
        parameter_dict[par.name] = np.frombuffer(
            par.raw_data, dtype=np.float32
        ).reshape(par.dims)

    for index, node in enumerate(graph.node):
        if index==len(graph.node)-1:
            for in_put in node.input:
                if in_put in parameter_dict.keys() and parameter_dict[in_put].ndim ==2:
                    arr = parameter_dict[in_put]
                    break


    if arr.ndim != 2:
        raise ValueError(f'The dimension of the last layer must be 2.ndmi:{arr.ndim},modelname:{model_path}')

    num_max = np.max(arr)
    num_min = np.min(arr)
    arr = (arr-num_min)/(num_max-num_min)  # minmax_norm

    colum_shape = arr.shape[1]
    bins = np.linspace(0, 1, args.bin_num)
    histograms = np.array([np.histogram(arr[idx, :], bins=bins)[0]
                  for idx in range(arr.shape[0])])
    arr = histograms/colum_shape  # bins for rate

    return (torch.tensor(arr, dtype=torch.float32), torch.tensor(is_poisoned, dtype=torch.long))
