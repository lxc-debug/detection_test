from torch.utils.data import Dataset
import torch
from utils.from_onnx_to_dgl import model2onnx, onnx2dgl
from config.conf import args, logger
import os
from glob import glob
from tqdm import tqdm
import re
import numpy as np


class MyDataset(Dataset):
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
                for model_path in tqdm(model_paths, desc=f'正在从{model_use}的{self.mode}的结构{args.architecture if args.use_archi else "all"}:onnx数据导入dataset'):
                    if 'clean' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 0
                    elif 'poisoned' in re.split(r'\.|/|_', model_path):
                        is_poisoned = 1
                    else:
                        raise NameError(f'文件命名错误,当前文件路径:{model_path}')

                    data_tuple = onnx2dgl(model_path, is_poisoned=is_poisoned)

                    self.data.append(data_tuple[0])
                    self.label.append(data_tuple[1])

            self._save(save_path)

        else:  # load
            if args.use_archi:
                save_path = os.path.join(
                    save_path, args.architecture, 'data.ds')
            else:
                save_path = os.path.join(save_path, 'data.ds')

            if not os.path.exists(save_path):
                raise ValueError('file not exists please process data')

            logger.info(f'载入{self.mode}数据')
            self._load(save_path)

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
            for name in tqdm(names, desc=f'将{dir.split("/")[-1]}中的模型转换为onnx'):
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
            for name in tqdm(names, desc=f'将{dir.split("/")[-1]}中的模型转换为onnx'):
                model2onnx(model, name, input_tensor, os.path.join(
                    self.save_dir, 'leader_two', self.mode))

    # 已经完成数据读取，后面再接着写就行
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

        return torch.tensor(np.array(data_li,dtype=np.float32), dtype=torch.float32), torch.tensor(np.array(label_li,np.int64), dtype=torch.long)
