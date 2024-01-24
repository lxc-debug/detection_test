import torch
import onnx
import numpy as np
from config.conf import args
import os
import dgl


def model2onnx(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    save_dir: str = args.onnx_model,
) -> None:
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

    # judge before loading
    if model is None:
        model = torch.load(model_name)
        model = model.cpu()
    else:
        stat = torch.load(model_name)
        model.load_state_dict(stat)

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
        if index == len(graph.node)-1:
            for in_put in node.input:
                if in_put in parameter_dict.keys() and parameter_dict[in_put].ndim == 2:
                    arr = parameter_dict[in_put]
                    break

    if arr.ndim != 2:
        raise ValueError(
            f'The dimension of the last layer must be 2.ndmi:{arr.ndim},modelname:{model_path}')

    if args.use_base:   # norm->mean->var for fairness also use 10 features
        final_res = list()
        for idx in range(arr.shape[0]):
            res = list()
            # origin
            arr_tmp = arr[idx, :]
            res.append(arr_tmp.min())
            res.append(arr_tmp.max())
            res.append(arr_tmp.mean())
            res.append(arr_tmp.var())
            res.append(arr.mean(axis=1).var())

            num_max = np.max(arr)
            num_min = np.min(arr)
            arr_norm = (arr-num_min)/(num_max-num_min)

            tmp = arr_norm[idx, :]
            # min max norm
            res.append(tmp.min())
            res.append(tmp.max())
            res.append(tmp.mean())
            res.append(tmp.var())
            res.append(arr_norm.mean(axis=1).var())

            arr_mean = np.mean(arr)
            arr_one_norm = np.greater_equal(
                arr, arr_mean.reshape(-1, 1)).astype(np.float32)
            tmp_one = arr_one_norm[idx, :]

            # zero one norm
            res.append(tmp_one.mean())
            res.append(tmp_one.var())
            res.append(arr_one_norm.mean(axis=1).var())

            if args.use_three:
                if model_path.split('/')[-1].split('_')[1]=='densenet121':
                    res.extend([1,0,0])
                elif model_path.split('/')[-1].split('_')[1]=='inceptionv3':
                    res.extend([0,1,0])
                elif model_path.split('/')[-1].split('_')[1]=='resnet50':
                    res.extend([0,0,1])
                else:
                    raise ValueError(f'model type not in list, the model_path is {model_path},and the type now is {model_path.split("/")[-1].split("_")[1]}')

            final_res.append(res)

        arr_final = np.array(final_res, dtype=np.float32)

    else:
        num_max = np.max(arr)
        num_min = np.min(arr)
        arr = (arr-num_min)/(num_max-num_min)  # minmax_norm

        colum_shape = arr.shape[1]
        bins = np.linspace(0, 1, args.bin_num)
        histograms = np.array([np.histogram(arr[idx, :], bins=bins)[0]
                               for idx in range(arr.shape[0])])
        arr_final = histograms/colum_shape  # bins for probability

        if args.use_three:
            row_num=arr_final.shape[0]
            if model_path.split('/')[-1].split('_')[1]=='densenet121':
                class_num=np.repeat(np.expand_dims(np.array([1,0,0]),axis=0),row_num,axis=0)
                arr_final=np.concatenate([arr_final,class_num],axis=1)
            elif model_path.split('/')[-1].split('_')[1]=='inceptionv3':
                class_num=np.repeat(np.expand_dims(np.array([0,1,0]),axis=0),row_num,axis=0)
                arr_final=np.concatenate([arr_final,class_num],axis=1)
            elif model_path.split('/')[-1].split('_')[1]=='resnet50':
                class_num=np.repeat(np.expand_dims(np.array([0,0,1]),axis=0),row_num,axis=0)
                arr_final=np.concatenate([arr_final,class_num],axis=1)
            else:
                raise ValueError(f'model type not in list, the model_path is {model_path},and the type now is {model_path.split("/")[-1].split("_")[1]}')

    # # 这里先用padding去解决问题了，但是后面感觉肯定不能这么整
    # arr=np.pad(arr, ((0, args.padding_dim-arr.shape[0]), (0, 0)), 'constant',constant_values=0)
    return arr_final, is_poisoned


def pos_emb(
    pos_array: np.ndarray, pos_emb_dim: int, embedding_type=args.embedding_type
):
    if embedding_type == "nerf":
        freq_band = np.linspace(
            2.0**0.0, 2.0 ** (pos_emb_dim - 1), num=pos_emb_dim)
        fn_list = []
        op_fn = [np.sin, np.cos]
        for freq in freq_band:
            for op in op_fn:
                fn_list.append(lambda x, freq=freq,
                               op=op: op(x * freq * np.pi))

    elif embedding_type == "trans":
        freq_band = [1 / (10000 ** (idx / pos_emb_dim))
                     for idx in range(pos_emb_dim)]
        fn_list = []
        op_fn = [np.sin, np.cos]
        for freq in freq_band:
            for op in op_fn:
                fn_list.append(lambda x, freq=freq, op=op: op(x * freq))

    return np.concatenate(
        [np.expand_dims(fn(pos_array), -1) for fn in fn_list], axis=-1
    )


def onnx2dgl2(
    model_path: str,
    is_poisoned: int = 1,
) -> tuple:
    """现在这个里面我自己想的是要返回三个部分，一个就是图，这个图里面有各种各样的属性，包括op_type，size还有位置，然后还有一个参数矩阵，应该是二维的，一维是row_size一维是提取到的特征参数，最后一个就是返回这个数据的标签，现在考虑的就是在标签之前应该返回一个列表，表示原本每个row的大小，因为准备的是row直接padding到指定的大小，"""
    onnx_model = onnx.load(model_path)

    graph = onnx_model.graph

    parameter_dict = dict()  # 网络参数

    for par in graph.initializer:  # 转换所有的网络参数
        parameter_dict[par.name] = np.frombuffer(
            par.raw_data, dtype=np.float32
        ).reshape(par.dims)

    par_li = list()
    graph_par_li = list()
    row_size = list()
    node_index = dict()
    edge_src = list()
    edge_dst = list()

    for index, node in enumerate(graph.node):
        # 首先是对参数的操作，然后是对结构的操作
        arr = None
        for in_put in node.input:
            row_arr = np.zeros((args.row_size, args.bin_num), dtype=np.float32)
            # 想了一下，这一版还是主要考虑w这个参数
            if in_put in parameter_dict.keys() and (in_put.split('.')[-1] == 'weight') or (in_put.startswith('onnx::Conv') and parameter_dict[in_put].ndim == 4):
                arr = parameter_dict[in_put]
                if arr.ndim == 4:  # 这个是conv的参数
                    row_shape, _, _, _ = arr.shape
                    arr = arr.reshape((row_shape, -1))

                if args.use_base:   # norm->mean->var for fairness also use 10 features
                    if arr.shape[0] >= args.row_size:
                        raise ValueError(
                            f'row size is out of setting, now needing space is {arr.shape[0]}')

                    num_max = np.max(arr)
                    num_min = np.min(arr)
                    # 把这个运算放到外面就只需要算一次了
                    arr_norm = (arr-num_min)/(num_max-num_min)

                    arr_mean = np.mean(arr)
                    arr_one_norm = np.greater_equal(
                        arr, arr_mean.reshape(-1, 1)).astype(np.float32)

                    for idx in range(arr.shape[0]):
                        # origin
                        arr_tmp = arr[idx, :]
                        row_arr[idx][0] = arr_tmp.min()
                        row_arr[idx][1] = arr_tmp.max()
                        row_arr[idx][2] = arr_tmp.mean()
                        row_arr[idx][3] = arr_tmp.var()
                        row_arr[idx][4] = arr.mean(axis=1).var()

                        tmp = arr_norm[idx, :]
                        # min max norm
                        row_arr[idx][5] = tmp.min()
                        row_arr[idx][6] = tmp.max()
                        row_arr[idx][7] = tmp.mean()
                        row_arr[idx][8] = tmp.var()
                        row_arr[idx][9] = arr_norm.mean(axis=1).var()

                        tmp_one = arr_one_norm[idx, :]

                        # zero one norm
                        row_arr[idx][10] = tmp_one.mean()
                        row_arr[idx][11] = tmp_one.var()
                        row_arr[idx][12] = arr_one_norm.mean(axis=1).var()

                else:
                    num_max = np.max(arr)
                    num_min = np.min(arr)
                    arr = (arr-num_min)/(num_max-num_min)  # minmax_norm

                    colum_shape = arr.shape[1]
                    bins = np.linspace(0, 1, args.bin_num)
                    histograms = np.array([np.histogram(arr[idx, :], bins=bins)[0]
                                           for idx in range(arr.shape[0])], dtype=np.float32)
                    arr_bin = histograms/colum_shape  # bins for probability
                    row_arr[:arr.shape[0], :] = arr_bin

            par_li.append(row_arr)

        if arr:
            row_size.append(arr.shape[0])
        else:
            row_size.append(0)

        # 下面是处理图的数据的
        op_type = np.zeros((args.op_type_size,), dtype=np.float32)
        op_index = args.op_li.index(node.op_type)
        op_type[op_index] = 1.

        node_size = np.zeros((4,), dtype=np.float32)
        if arr is not None:
            if arr.ndim == 4:
                node_size += np.array(arr.shape, dtype=np.float32)
            elif arr.ndim == 2:
                node_size[2:] += np.array(arr.shape, dtype=np.float32)
        node_size = pos_emb(node_size, args.pos_emb_dim).reshape((-1,))

        pos = pos_emb(np.array([index], dtype=np.float32),
                      args.pos_emb_dim).reshape((-1,))

        # 现在还缺的就是一个pos_emb，然后把这些东西展开拼起来，最后还有就是mask的大小还没有确定。
        struct_arr = np.concatenate((op_type, node_size, pos))
        graph_par_li.append(struct_arr)

        # 下面是处理结构的连边关系的
        for out in node.output:  # 得到节点输出用来图的连接
            node_index[out] = index

        for in_put in node.input:
            if in_put in node_index.keys():
                edge_src.append(node_index[in_put])
                edge_dst.append(index)

    node_size = len(list(graph.node))
    if args.num_nodes < node_size:
        raise ValueError(f'node size seeting small, needing {node_size}')

    for _ in range(args.num_nodes-node_size):
        row_size.append(0)

    # 现在想的是这里直接把mask算好就完了，不在模型的forward中算了，不然有点麻烦
    row_mask=np.zeros((args.num_nodes,row_size),np.float32)
    for idx,r_size in enumerate(row_size):
        row_mask[idx,:r_size]=1.
    row_mask=np.expand_dims(row_mask,axis=1)    # 这里的shape就是(ns,1,rs)

    dgl_graph = dgl.to_bidirected(
        dgl.graph((edge_src, edge_dst), num_nodes=args.num_nodes))

    graph_par_tensor = torch.FloatTensor(graph_par_li)
    par_tensor = torch.FloatTensor(par_li)

    dgl_graph.ndata['struct_feature'] = graph_par_tensor

    node_mask=np.zeros((args.node_size,),dtype=np.float32)
    node_mask[:node_size]=1.
    node_mask=np.expand_dims(node_mask,axis=1)

    return dgl_graph, par_tensor, node_mask, row_mask
    # 读数据的部分大体上完成了，明天来了把网络的实现再改了就ok了
