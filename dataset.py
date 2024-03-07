from model.resnet import resnet18_mod
from model.vgg_model import CNN_classifier
import numpy as np
import re
from tqdm import tqdm
from glob import glob
import os
from config.log_conf import logger
from config.conf import args
from utils.from_onnx_to_dgl import model2onnx, onnx2dgl, onnx2dgltest, onnx2dgl_posemb_test, onnx2dgl2,onnx2dgl_pos_main
import torch
from torch.utils.data import Dataset
import dgl
import pickle
import sys
sys.path.append('./model')


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def _trans_vgg_model(self):
        model = CNN_classifier()
        input_tensor = torch.randn(1, 3, 32, 32)

        if self.mode == 'train':
            dir_list = args.vgg_train_data_dir
        elif self.mode == 'eval':
            dir_list = args.vgg_eval_data_dir
        elif self.mode == 'test':
            dir_list = args.vgg_test_data_dir
        else:
            raise ValueError('mode must be train or eval or test')

        for dir in dir_list:
            names = glob(os.path.join(dir, '*.pt'))
            for name in tqdm(names,):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'vgg', self.mode))

    def _trans_resnet_model(self):
        model = resnet18_mod(num_classes=200)  
        input_tensor = torch.randn(1, 3, 32, 32)

        if self.mode == 'train':
            dir_list = args.resnet_train_data_dir
        elif self.mode == 'eval':
            dir_list = args.resnet_eval_data_dir
        elif self.mode == 'test':
            dir_list = args.resnet_test_data_dir
        else:
            raise ValueError('mode must be train or eval or test')

        for dir in dir_list:
            names = glob(os.path.join(dir, '*.pt'))
            for name in tqdm(names, ):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'resnet', self.mode))

    def _trans_leader_one_model(self):
        model = None
        input_tensor = torch.randn(1, 3, 224, 224)

        if self.mode == 'train':
            dir_list = args.leaderone_train_data_dir
        elif self.mode == 'eval':
            dir_list = args.leaderone_eval_data_dir
        elif self.mode == 'test':
            dir_list = args.leaderone_test_data_dir
        else:
            raise ValueError('mode must be train or eval or test')

        for dir in dir_list:
            names = glob(os.path.join(dir, '*.pt'))
            for name in tqdm(names,):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'leader_one', self.mode))

    def _trans_leader_two_model(self):
        model = None
        input_tensor = torch.randn(1, 3, 224, 224)

        if self.mode == 'train':
            dir_list = args.leadertwo_train_data_dir
        elif self.mode == 'eval':
            dir_list = args.leadertwo_eval_data_dir
        elif self.mode == 'test':
            dir_list = args.leadertwo_test_data_dir
        else:
            raise ValueError('mode must be train or eval or test')

        for dir in dir_list:
            names = glob(os.path.join(dir, '*.pt'))
            for name in tqdm(names,):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'leader_two', self.mode))
                
    def _trans_leader_three_model(self):
        model = None
        input_tensor = torch.randn(1, 3, 224, 224)

        if self.mode == 'train':
            dir_list = args.leaderthree_train_data_dir
        elif self.mode == 'eval':
            dir_list = args.leaderthree_eval_data_dir
        elif self.mode == 'test':
            dir_list = args.leaderthree_test_data_dir
        else:
            raise ValueError('mode must be train or eval or test')

        for dir in dir_list:
            names = glob(os.path.join(dir, '*.pt'))
            for name in tqdm(names,):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'leader_three', self.mode))
                

    def _trans_leader_four_model(self):
        model = None
        input_tensor = torch.randn(1, 3, 224, 224)

        if self.mode == 'train':
            dir_list = args.leaderfour_train_data_dir
        elif self.mode == 'eval':
            dir_list = args.leaderfour_eval_data_dir
        elif self.mode == 'test':
            dir_list = args.leaderfour_test_data_dir
        else:
            raise ValueError('mode must be train or eval or test')

        for dir in dir_list:
            names = glob(os.path.join(dir, '*.pt'))
            for name in tqdm(names,):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'leader_four', self.mode))


class MyDataset(BaseDataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data = list()
        self.label = list()
        self.mode = mode
        self.use_list = args.use_list
        self.save_dir = args.onnx_model

        if args.load_parameter:
            for model_use in self.use_list:
                if model_use == 'leader_one':
                    self._trans_leader_one_model()
                elif model_use == 'leader_two':
                    self._trans_leader_two_model()
                if model_use == 'vgg':
                    self._trans_vgg_model()
                if model_use == 'resnet':
                    self._trans_resnet_model()

        file_name = '+'.join(self.use_list)
        save_path = os.path.join(args.data_save, file_name, self.mode)
        if args.process_data:  # save
            for model_use in self.use_list:
                if args.use_archi:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, args.architecture, '*.onnx')
                else:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, '*.onnx')

                model_paths = glob(model_paths)

                for model_path in tqdm(model_paths):
                    if 'clean' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 0
                    elif 'poisoned' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 1
                    else:
                        raise NameError(f'{model_path}')

                    data = onnx2dgl(model_path)

                    self.data.append(data)
                    self.label.append(is_poisoned)

            self._save(save_path)

        else:  # load
            if args.use_archi:
                save_path = os.path.join(
                    save_path, args.architecture, 'data.ds')
            else:
                save_path = os.path.join(save_path, 'data.ds')

            if not os.path.exists(save_path):
                raise ValueError('file not exists please process data')

            self._load(save_path)

    def _load(self, save_path):
        self.data, self.label = torch.load(save_path)

    def _save(self, save_path):
        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, 'data.ds')
        else:
            save_path = os.path.join(save_path, 'data.ds')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        torch.save((self.data, self.label), save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.label[index]

    @staticmethod
    def coll_fn(batch):
        max_len = max(batch, key=lambda x: x[0].shape[0])[0].shape[0]

        data_li = list()
        label_li = list()

        for data, label in batch:
            data = np.pad(
                data, ((0, max_len-data.shape[0]), (0, 0)), 'constant', constant_values=0)
            data_li.append(data)
            label_li.append(label)

        return torch.tensor(np.array(data_li, dtype=np.float32), dtype=torch.float32), torch.tensor(np.array(label_li, np.int64), dtype=torch.long)


class TestDataset(BaseDataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data = list()
        self.label = list()
        self.row_mask = list()
        self.node_mask = list()
        self.archi = list()
        self.mode = mode
        self.use_list = args.use_list
        self.save_dir = args.onnx_model

        if args.load_parameter:
            for model_use in self.use_list:
                if model_use == 'leader_one':
                    self._trans_leader_one_model()
                elif model_use == 'leader_two':
                    self._trans_leader_two_model()
                if model_use == 'vgg':
                    self._trans_vgg_model()
                if model_use == 'resnet':
                    self._trans_resnet_model()

        file_name = '+'.join(self.use_list)
        save_path = os.path.join(args.data_save, file_name, self.mode)
        if args.process_data:  # save
            for model_use in self.use_list:
                if args.use_archi:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, args.architecture, '*.onnx')
                else:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, '*.onnx')

                model_paths = glob(model_paths)

                for model_path in tqdm(model_paths):
                    if 'clean' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 0
                    elif 'poisoned' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 1
                    else:
                        raise NameError(f'{model_path}')

                    data_res = onnx2dgltest(model_path)

                    self.data.append(data_res[0])
                    self.row_mask.append(data_res[1])
                    self.node_mask.append(data_res[2])
                    self.archi.append(data_res[3])
                    self.label.append(is_poisoned)

            # self._save(save_path)

        else:  # load
            if args.use_base:
                file_name='base_data.ds'
            else :
                file_name='bin_data.ds'    
            
            if args.use_archi:
                save_path = os.path.join(
                    save_path, args.architecture, file_name)
            else:
                save_path = os.path.join(save_path, file_name)

            if not os.path.exists(save_path):
                raise ValueError('file not exists please process data')

            self._load(save_path)

    def _load(self, save_path):
        self.data, self.row_mask, self.node_mask, self.label, self.archi = torch.load(
            save_path)

    def _save(self, save_path):
        if args.use_base:
            file_name='base_data.ds'
        else :
            file_name='bin_data.ds'

        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, file_name)
        else:
            save_path = os.path.join(save_path, file_name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        torch.save((self.data, self.row_mask,
                   self.node_mask, self.label, self.archi), save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.row_mask[index], self.node_mask[index], self.label[index], self.archi[index]

    @staticmethod
    def coll_fn(batch):
        data_li = list()
        label_li = list()
        row_mask_li = list()
        node_mask_li = list()
        archi_li = list()

        for data, row_mask, nodemask, label, archi in batch:
            if not isinstance(data,torch.Tensor):
                data = torch.tensor(data,dtype=torch.float32)
            if not isinstance(row_mask,torch.Tensor):  
                row_mask = torch.tensor(row_mask,dtype=torch.float32)
            if not isinstance(nodemask,torch.Tensor):
                nodemask = torch.tensor(nodemask,dtype=torch.float32)
            if not isinstance(label,torch.Tensor):
                label = torch.tensor(label,dtype=torch.long)
            if not isinstance(archi,torch.Tensor):
                archi = torch.tensor(archi,dtype=torch.float32)
            data_li.append(data)
            label_li.append(label)
            row_mask_li.append(row_mask)
            node_mask_li.append(nodemask)
            archi_li.append(archi)

        return torch.stack(data_li,dim=0), torch.stack(row_mask_li,dim=0), torch.stack(node_mask_li,dim=0), torch.stack(label_li,dim=0), torch.stack(archi_li,dim=0)
    

class PosEmbTestDataset(BaseDataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data = list()
        self.label = list()
        self.row_mask = list()
        self.node_mask = list()
        self.archi = list()
        self.pos_emb = list()
        self.mode = mode
        self.use_list = args.use_list
        self.save_dir = args.onnx_model

        if args.load_parameter:
            for model_use in self.use_list:
                if model_use == 'leader_one':
                    self._trans_leader_one_model()
                elif model_use == 'leader_two':
                    self._trans_leader_two_model()
                if model_use == 'vgg':
                    self._trans_vgg_model()
                if model_use == 'resnet':
                    self._trans_resnet_model()

        file_name = '+'.join(self.use_list)
        save_path = os.path.join(args.data_save, file_name, self.mode)
        if args.process_data:  # save
            for model_use in self.use_list:
                if args.use_archi:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, args.architecture, '*.onnx')
                else:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, '*.onnx')

                model_paths = glob(model_paths)

                for model_path in tqdm(model_paths,):
                    if 'clean' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 0
                    elif 'poisoned' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 1
                    else:
                        raise NameError(f'{model_path}')

                    data_res = onnx2dgl_posemb_test(model_path)

                    self.data.append(data_res[0])
                    self.row_mask.append(data_res[1])
                    self.node_mask.append(data_res[2])
                    self.archi.append(data_res[3])
                    self.pos_emb.append(data_res[4])
                    self.label.append(is_poisoned)

            self._save(save_path)

        else:  # load
            if args.use_base:
                file_name='base_data.ds'
            else :
                file_name='bin_data.ds'
            if args.use_archi:
                save_path = os.path.join(
                    save_path, args.architecture, file_name)
            else:
                save_path = os.path.join(save_path, file_name)

            if not os.path.exists(save_path):
                raise ValueError('file not exists please process data')

            self._load(save_path)

    def _load(self, save_path):
        self.data, self.row_mask, self.node_mask, self.label, self.archi, self.pos_emb = torch.load(
            save_path)

    def _save(self, save_path):
        if args.use_base:
            file_name='base_data.ds'
        else :
            file_name='bin_data.ds'

        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, file_name)
        else:
            save_path = os.path.join(save_path, file_name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        torch.save((self.data, self.row_mask,
                   self.node_mask, self.label, self.archi, self.pos_emb), save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.row_mask[index], self.node_mask[index], self.label[index], self.archi[index], self.pos_emb[index]

    @staticmethod
    def coll_fn(batch):
        data_li = list()
        label_li = list()
        row_mask_li = list()
        node_mask_li = list()
        archi_li = list()
        pos_emb_li = list()

        for data, row_mask, nodemask, label, archi, pos_emb in batch:
            data = torch.tensor(data,dtype=torch.float32)
            row_mask = torch.tensor(row_mask,dtype=torch.float32)
            nodemask = torch.tensor(nodemask,dtype=torch.float32)
            label = torch.tensor(label,dtype=torch.long)
            archi = torch.tensor(archi,dtype=torch.float32)
            pos_emb = torch.tensor(pos_emb,dtype=torch.float32)
            
            data_li.append(data)
            label_li.append(label)
            row_mask_li.append(row_mask)
            node_mask_li.append(nodemask)
            archi_li.append(archi)
            pos_emb_li.append(pos_emb)

        return torch.stack(data_li,dim=0), torch.stack(row_mask_li,dim=0), torch.stack(node_mask_li,dim=0), torch.stack(label_li,dim=0), torch.stack(archi_li,dim=0), torch.stack(pos_emb_li,dim=0)



class MainDataset(BaseDataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data = list()
        self.label = list()
        self.row_mask = list()
        self.node_mask = list()
        self.graph_list = list()
        self.pos_emb = list()
        self.mode = mode
        self.use_list = args.use_list
        self.save_dir = args.onnx_model

        if args.load_parameter:
            for model_use in self.use_list:
                if model_use == 'leader_one':
                    self._trans_leader_one_model()
                elif model_use == 'leader_two':
                    self._trans_leader_two_model()
                elif model_use == 'leader_three':
                    self._trans_leader_three_model()
                elif model_use == 'leader_four':
                    self._trans_leader_four_model()
                elif model_use == 'vgg':
                    self._trans_vgg_model()
                elif model_use == 'resnet':
                    self._trans_resnet_model()

        file_name = '+'.join(self.use_list)
        save_path = os.path.join(args.data_save, file_name, self.mode)
        if args.process_data:  # save
            for model_use in self.use_list:
                if args.use_archi:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, args.architecture, '*.onnx')
                else:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, '*.onnx')

                model_paths = glob(model_paths)

                for model_path in tqdm(model_paths):
                    if 'clean' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 0
                    elif 'poisoned' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 1
                    else:
                        raise NameError(f'{model_path}')

                    data_res = onnx2dgl2(model_path)

                    self.data.append(data_res[1])
                    self.row_mask.append(data_res[2])
                    self.node_mask.append(data_res[3])
                    self.graph_list.append(data_res[0])
                    # self.pos_emb.append(data_res[4])
                    self.label.append(is_poisoned)

            # self._save(save_path)
            # pass

        else:  # load
            self._load(save_path)

    def _load(self, save_path):

        if args.use_base:
            file_name='base_data.ds'
        else :
            file_name='bin_data.ds'
        
        if args.main:
            file_name='main_'+file_name


        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, file_name)
        else:
            save_path = os.path.join(save_path, file_name)


        if not os.path.exists(save_path):
            raise ValueError('file not exists please process data')

        # self.data, self.row_mask, self.node_mask, self.label = torch.load(
        #     save_path)
        # self.graph_list=torch.load(graph_save_path)
        with open(save_path,'rb') as fp:
            load_content=pickle.load(fp)

        self.graph_list=load_content[0]
        self.data=load_content[1]
        self.row_mask=load_content[2]
        self.node_mask=load_content[3]
        self.label=load_content[4]


    def _save(self, save_path):
        if args.use_base:
            file_name='base_data.ds'
        else :
            file_name='bin_data.ds'

        if args.main:
            file_name='main_'+file_name


        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, file_name)
        else:
            save_path = os.path.join(save_path, file_name)


        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))


        # torch.save((self.data, self.row_mask,
        #            self.node_mask, self.label), save_path)
        # torch.save(self.graph_list,graph_save_path)
        with open(save_path,'wb') as fp:
            pickle.dump((self.graph_list, self.data, self.row_mask,
                   self.node_mask, self.label),fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        return self.graph_list[index], self.data[index], self.row_mask[index], self.node_mask[index], self.label[index]

    @staticmethod
    def coll_fn(batch):
        data_li = list()
        label_li = list()
        row_mask_li = list()
        node_mask_li = list()
        graph_li = list()
        

        for graph, data, row_mask, node_mask, label in batch:
            data = torch.tensor(data,dtype=torch.float32)
            row_mask = torch.tensor(row_mask,dtype=torch.float32)
            node_mask = torch.tensor(node_mask,dtype=torch.float32)
            label = torch.tensor(label,dtype=torch.long)
            
            
            data_li.append(data)
            label_li.append(label)
            row_mask_li.append(row_mask)
            node_mask_li.append(node_mask)
            graph_li.append(graph)

            

        return dgl.batch(graph_li), torch.stack(data_li,dim=0), torch.stack(row_mask_li,dim=0), torch.stack(node_mask_li,dim=0), torch.stack(label_li,dim=0)
    

class PosMainDataset(BaseDataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data = list()
        self.label = list()
        self.row_mask = list()
        self.node_mask = list()
        self.graph_list = list()
        self.pos_emb = list()
        self.mode = mode
        self.use_list = args.use_list
        self.save_dir = args.onnx_model

        if args.load_parameter:
            for model_use in self.use_list:
                if model_use == 'leader_one':
                    self._trans_leader_one_model()
                elif model_use == 'leader_two':
                    self._trans_leader_two_model()
                elif model_use == 'leader_three':
                    self._trans_leader_three_model()
                elif model_use == 'leader_four':
                    self._trans_leader_four_model()
                elif model_use == 'vgg':
                    self._trans_vgg_model()
                elif model_use == 'resnet':
                    self._trans_resnet_model()

        file_name = '+'.join(self.use_list)
        save_path = os.path.join(args.data_save, file_name, self.mode)
        if args.process_data:  # save
            for model_use in self.use_list:
                if args.use_archi:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, args.architecture, '*.onnx')
                else:
                    model_paths = os.path.join(
                        self.save_dir, model_use, self.mode, '*.onnx')

                model_paths = glob(model_paths)

                for model_path in tqdm(model_paths):
                    if 'clean' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 0
                    elif 'poisoned' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 1
                    else:
                        raise NameError(f'{model_path}')

                    data_res = onnx2dgl_pos_main(model_path)

                    self.graph_list.append(data_res[0])
                    self.data.append(data_res[1])
                    self.row_mask.append(data_res[2])
                    self.node_mask.append(data_res[3])
                    self.pos_emb.append(data_res[4])
                    self.label.append(is_poisoned)

            # self._save(save_path)
            # pass

        else:  # load
            self._load(save_path)

    def _load(self, save_path):

        if args.use_base:
            file_name='base_data.ds'
        else :
            file_name='bin_data.ds'
        
        if args.main:
            file_name='main_'+file_name


        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, file_name)
        else:
            save_path = os.path.join(save_path, file_name)


        if not os.path.exists(save_path):
            raise ValueError('file not exists please process data')

        # self.data, self.row_mask, self.node_mask, self.label = torch.load(
        #     save_path)
        # self.graph_list=torch.load(graph_save_path)
        with open(save_path,'rb') as fp:
            load_content=pickle.load(fp)

        self.graph_list=load_content[0]
        self.data=load_content[1]
        self.row_mask=load_content[2]
        self.node_mask=load_content[3]
        self.pos_emb=load_content[4]
        self.label=load_content[5]


    def _save(self, save_path):
        if args.use_base:
            file_name='base_data.ds'
        else :
            file_name='bin_data.ds'

        if args.main:
            file_name='main_'+file_name


        if args.use_archi:
            save_path = os.path.join(
                save_path, args.architecture, file_name)
        else:
            save_path = os.path.join(save_path, file_name)


        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))


        # torch.save((self.data, self.row_mask,
        #            self.node_mask, self.label), save_path)
        # torch.save(self.graph_list,graph_save_path)
        with open(save_path,'wb') as fp:
            pickle.dump((self.graph_list, self.data, self.row_mask,
                   self.node_mask, self.pos_emb, self.label),fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        return self.graph_list[index], self.data[index], self.row_mask[index], self.node_mask[index], self.pos_emb[index], self.label[index]

    @staticmethod
    def coll_fn(batch):
        data_li = list()
        label_li = list()
        row_mask_li = list()
        node_mask_li = list()
        graph_li = list()
        pos_emb_li = list()
        

        for graph, data, row_mask, node_mask, pos_emb, label in batch:
            data = torch.tensor(data,dtype=torch.float32)
            row_mask = torch.tensor(row_mask,dtype=torch.float32)
            node_mask = torch.tensor(node_mask,dtype=torch.float32)
            label = torch.tensor(label,dtype=torch.long)
            pos_emb = torch.tensor(pos_emb,dtype=torch.float32)
            
            
            data_li.append(data)
            label_li.append(label)
            row_mask_li.append(row_mask)
            node_mask_li.append(node_mask)
            graph_li.append(graph)
            pos_emb_li.append(pos_emb)

            

        return dgl.batch(graph_li), torch.stack(data_li,dim=0), torch.stack(row_mask_li,dim=0), torch.stack(node_mask_li,dim=0), torch.stack(pos_emb_li,dim=0), torch.stack(label_li,dim=0)