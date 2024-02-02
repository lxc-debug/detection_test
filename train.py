from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from config.conf import args
from config.log_conf import logger, file_logger
import os
from sklearn.metrics import roc_auc_score
import numpy as np


class Experiment():
    def __init__(self, model: nn.Module, train_dataset, eval_dataset, test_dataset) -> None:
        """_summary_
            准备训练用到的dataloader，模型以及优化器，还有一些日志的保存路径
        Arguments:
            model -- 待训练的模型
            train_dataset -- 训练数据集
            eval_dataset -- 验证数据集
            test_dataset -- 测试数据集
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.optim = optim.Adam(self.model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)
        self.epochs = args.epochs
        self.loss_fn = torch.nn.CrossEntropyLoss()

        filename = '+'.join(args.use_list)
        if args.use_archi:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, args.architecture, 'best.pt')
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)
        else:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, 'best.pt')
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)

        self.tensorboard_dir = os.path.join(
            tensorboard_dir, f'lr:{args.lr}_wd:{args.weight_decay}')
        self.esp = EarlyStopping(self.model, self.save_dir)

    def _train(self, epoch):
        """_summary_
            训练模型
        Arguments:
            epoch -- 第几次训练
        """
        self.model.train()

        self.total_correct = 0
        self.total_loss = 0

        t_bar = tqdm(total=len(self.train_dataloader), desc=f'第{epoch+1}轮训练')
        for data, label in self.train_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)

            res = self.model(data)
            loss = self.loss_fn(res, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            correct = (torch.argmax(res, dim=-1) == label).sum().item()
            self.total_correct += correct
            self.total_loss += loss.item()*res.shape[0]

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _eval(self, epoch):
        """_summary_
            模型验证
        Arguments:
            epoch -- 第几次验证
        """
        self.model.eval()

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        t_bar = tqdm(total=len(self.eval_dataloader), desc=f'第{epoch+1}轮验证')
        for data, label in self.eval_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)

            res = self.model(data)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _test(self):
        """_summary_
            模型测试
        """
        self.model.eval()

        self.pre_label = list()
        self.label = list()

        self.total_test_correct = 0
        self.total_test_loss = 0

        for data, label in tqdm(self.test_dataloader, desc='测试集分析'):
            data = data.to(self.device)
            label = label.to(self.device)

            res = self.model(data)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_test_correct += correct
            self.total_test_loss += loss.item()*res.shape[0]

            self.pre_label.extend(
                torch.nn.functional.softmax(res, dim=-1)[:, 1].tolist())
            self.label.extend(label.tolist())

        acc = self.total_test_correct/len(self.test_dataset)
        loss = self.total_test_loss/len(self.test_dataset)
        score = roc_auc_score(np.array(self.label), np.array(self.pre_label))

        logger.info(
            f'test dataset acc:{acc:8.4f}|loss:{loss:8.4f}|auc_roc_score:{score:8.6f}')

    def __call__(self) -> None:
        """_summary_
            整个模型的训练过程，首先是一轮训练和验证，结束后判断是否早停，并对日志进行记录和保存
        """
        self.writer = SummaryWriter(self.tensorboard_dir)
        for epoch in range(self.epochs):
            self._train(epoch=epoch)
            self._eval(epoch=epoch)

            self.esp(self.total_eval_loss/len(self.eval_dataset))
            if self.esp.break_now:
                print('early stopping...')
                break

            self.writer.add_scalar(
                'loss_epoch/train', self.total_loss/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/train', self.total_correct/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'loss_epoch/eval', self.total_eval_loss/len(self.eval_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/eval', self.total_eval_correct/len(self.eval_dataset), global_step=epoch)

            file_logger.info(f'{args.architecture if args.use_archi else "all":>10s}|epoch:{epoch:4d}|train_loss:{self.total_loss/len(self.train_dataset):8.4f}|train_acc:{self.total_correct/len(self.train_dataset):8.4f}|eval_loss:{self.total_eval_loss/len(self.eval_dataset):8.4f}|eval_acc:{self.total_eval_correct/len(self.eval_dataset):8.4f}')

        self.writer.close()
        self.model.load_state_dict(torch.load(self.save_dir))
        self._test()


class ExperimentTest():
    def __init__(self, model: nn.Module, train_dataset, eval_dataset, test_dataset) -> None:
        """_summary_
            准备训练用到的dataloader，模型以及优化器，还有一些日志的保存路径
        Arguments:
            model -- 待训练的模型
            train_dataset -- 训练数据集
            eval_dataset -- 验证数据集
            test_dataset -- 测试数据集
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.optim = optim.Adam(self.model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)
        self.epochs = args.epochs
        self.loss_fn = torch.nn.CrossEntropyLoss()

        filename = '+'.join(args.use_list)

        if args.use_base:
            save_file_name='best_base.pt'
        else:
            save_file_name='best_bin.pt'

        if args.use_archi:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, args.architecture, save_file_name)
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)
        else:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, save_file_name)
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)

        self.tensorboard_dir = os.path.join(
            tensorboard_dir, f'lr:{args.lr}_wd:{args.weight_decay}')
        self.esp = EarlyStopping(self.model, self.save_dir)

    def _train(self, epoch):
        """_summary_
            训练模型
        Arguments:
            epoch -- 第几次训练
        """
        self.model.train()

        self.total_correct = 0
        self.total_loss = 0

        t_bar = tqdm(total=len(self.train_dataloader), desc=f'第{epoch+1}轮训练')
        for data, row_mask, node_mask, label, archi in self.train_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)

            res = self.model(data, archi, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            correct = (torch.argmax(res, dim=-1) == label).sum().item()
            self.total_correct += correct
            self.total_loss += loss.item()*res.shape[0]

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _eval(self, epoch):
        """_summary_
            模型验证
        Arguments:
            epoch -- 第几次验证
        """
        self.model.eval()

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        t_bar = tqdm(total=len(self.eval_dataloader), desc=f'第{epoch+1}轮验证')
        for data, row_mask, node_mask, label, archi in self.eval_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)

            res = self.model(data, archi, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _test(self):
        """_summary_
            模型测试
        """
        self.model.eval()

        self.pre_label = list()
        self.label = list()

        self.total_test_correct = 0
        self.total_test_loss = 0

        for data, row_mask, node_mask, label, archi in tqdm(self.test_dataloader, desc='测试集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)

            res = self.model(data, archi, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_test_correct += correct
            self.total_test_loss += loss.item()*res.shape[0]

            self.pre_label.extend(
                torch.nn.functional.softmax(res, dim=-1)[:, 1].tolist())
            self.label.extend(label.tolist())

        self.total_correct = 0
        self.total_loss = 0
        for data, row_mask, node_mask, label, archi in tqdm(self.train_dataloader,desc='训练集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)

            res = self.model(data, archi, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_correct += correct

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        for data, row_mask, node_mask, label, archi in tqdm(self.eval_dataloader,desc='验证集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)

            res = self.model(data, archi, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

        acc = self.total_test_correct/len(self.test_dataset)
        loss = self.total_test_loss/len(self.test_dataset)
        score = roc_auc_score(np.array(self.label), np.array(self.pre_label))

        train_acc=self.total_correct/len(self.train_dataset)
        eval_acc=self.total_eval_correct/len(self.eval_dataset)

        logger.info(
            f'train_acc:{train_acc:8.4f}|eval_acc={eval_acc:8.4f}|test dataset acc:{acc:8.4f}|loss:{loss:8.4f}|auc_roc_score:{score:8.6f}')

    def __call__(self) -> None:
        """_summary_
            整个模型的训练过程，首先是一轮训练和验证，结束后判断是否早停，并对日志进行记录和保存
        """
        self.writer = SummaryWriter(self.tensorboard_dir)
        for epoch in range(self.epochs):
            self._train(epoch=epoch)
            self._eval(epoch=epoch)

            self.esp(self.total_eval_loss/len(self.eval_dataset))
            if self.esp.break_now:
                print('early stopping...')
                break

            self.writer.add_scalar(
                'loss_epoch/train', self.total_loss/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/train', self.total_correct/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'loss_epoch/eval', self.total_eval_loss/len(self.eval_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/eval', self.total_eval_correct/len(self.eval_dataset), global_step=epoch)

            file_logger.info(f'{args.architecture if args.use_archi else "all":>10s}|epoch:{epoch:4d}|train_loss:{self.total_loss/len(self.train_dataset):8.4f}|train_acc:{self.total_correct/len(self.train_dataset):8.4f}|eval_loss:{self.total_eval_loss/len(self.eval_dataset):8.4f}|eval_acc:{self.total_eval_correct/len(self.eval_dataset):8.4f}')

        self.writer.close()
        self.model.load_state_dict(torch.load(self.save_dir))
        self._test()



class ExperimentPosEmbTest():
    def __init__(self, model: nn.Module, train_dataset, eval_dataset, test_dataset) -> None:
        """_summary_
            准备训练用到的dataloader，模型以及优化器，还有一些日志的保存路径
        Arguments:
            model -- 待训练的模型
            train_dataset -- 训练数据集
            eval_dataset -- 验证数据集
            test_dataset -- 测试数据集
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.optim = optim.Adam(self.model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)
        self.epochs = args.epochs
        self.loss_fn = torch.nn.CrossEntropyLoss()

        filename = '+'.join(args.use_list)

        if args.use_base:
            save_file_name='best_base.pt'
        else:
            save_file_name='best_bin.pt'

        if args.use_archi:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, args.architecture, save_file_name)
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)
        else:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, save_file_name)
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)

        self.tensorboard_dir = os.path.join(
            tensorboard_dir, f'lr:{args.lr}_wd:{args.weight_decay}')
        self.esp = EarlyStopping(self.model, self.save_dir)

    def _train(self, epoch):
        """_summary_
            训练模型
        Arguments:
            epoch -- 第几次训练
        """
        self.model.train()

        self.total_correct = 0
        self.total_loss = 0

        t_bar = tqdm(total=len(self.train_dataloader), desc=f'第{epoch+1}轮训练')
        for data, row_mask, node_mask, label, archi, pos_emb in self.train_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)
            pos_emb = pos_emb.to(self.device)

            res = self.model(data, archi, row_mask, node_mask, pos_emb)
            loss = self.loss_fn(res, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            correct = (torch.argmax(res, dim=-1) == label).sum().item()
            self.total_correct += correct
            self.total_loss += loss.item()*res.shape[0]

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _eval(self, epoch):
        """_summary_
            模型验证
        Arguments:
            epoch -- 第几次验证
        """
        self.model.eval()

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        t_bar = tqdm(total=len(self.eval_dataloader), desc=f'第{epoch+1}轮验证')
        for data, row_mask, node_mask, label, archi, pos_emb in self.eval_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)
            pos_emb = pos_emb.to(self.device)

            res = self.model(data, archi, row_mask, node_mask, pos_emb)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _test(self):
        """_summary_
            模型测试
        """
        self.model.eval()

        self.pre_label = list()
        self.label = list()

        self.total_test_correct = 0
        self.total_test_loss = 0

        for data, row_mask, node_mask, label, archi, pos_emb in tqdm(self.test_dataloader, desc='测试集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)
            pos_emb = pos_emb.to(self.device)

            res = self.model(data, archi, row_mask, node_mask, pos_emb)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_test_correct += correct
            self.total_test_loss += loss.item()*res.shape[0]

            self.pre_label.extend(
                torch.nn.functional.softmax(res, dim=-1)[:, 1].tolist())
            self.label.extend(label.tolist())

        self.total_correct = 0
        self.total_loss = 0
        for data, row_mask, node_mask, label, archi, pos_emb in tqdm(self.train_dataloader,desc='训练集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)
            pos_emb = pos_emb.to(self.device)

            res = self.model(data, archi, row_mask, node_mask, pos_emb)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_correct += correct

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        for data, row_mask, node_mask, label, archi, pos_emb in tqdm(self.eval_dataloader,desc='验证集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            archi = archi.to(self.device)
            pos_emb = pos_emb.to(self.device)

            res = self.model(data, archi, row_mask, node_mask, pos_emb)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

        acc = self.total_test_correct/len(self.test_dataset)
        loss = self.total_test_loss/len(self.test_dataset)
        score = roc_auc_score(np.array(self.label), np.array(self.pre_label))

        train_acc=self.total_correct/len(self.train_dataset)
        eval_acc=self.total_eval_correct/len(self.eval_dataset)

        logger.info(
            f'train_acc:{train_acc:8.4f}|eval_acc={eval_acc:8.4f}|test dataset acc:{acc:8.4f}|loss:{loss:8.4f}|auc_roc_score:{score:8.6f}')

    def __call__(self) -> None:
        """_summary_
            整个模型的训练过程，首先是一轮训练和验证，结束后判断是否早停，并对日志进行记录和保存
        """
        self.writer = SummaryWriter(self.tensorboard_dir)
        for epoch in range(self.epochs):
            self._train(epoch=epoch)
            self._eval(epoch=epoch)

            self.esp(self.total_eval_loss/len(self.eval_dataset))
            if self.esp.break_now:
                print('early stopping...')
                break

            self.writer.add_scalar(
                'loss_epoch/train', self.total_loss/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/train', self.total_correct/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'loss_epoch/eval', self.total_eval_loss/len(self.eval_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/eval', self.total_eval_correct/len(self.eval_dataset), global_step=epoch)

            file_logger.info(f'{args.architecture if args.use_archi else "all":>10s}|epoch:{epoch:4d}|train_loss:{self.total_loss/len(self.train_dataset):8.4f}|train_acc:{self.total_correct/len(self.train_dataset):8.4f}|eval_loss:{self.total_eval_loss/len(self.eval_dataset):8.4f}|eval_acc:{self.total_eval_correct/len(self.eval_dataset):8.4f}')

        self.writer.close()
        self.model.load_state_dict(torch.load(self.save_dir))
        self._test()




class MainExperiment():
    def __init__(self, model: nn.Module, train_dataset, eval_dataset, test_dataset) -> None:
        """_summary_
            准备训练用到的dataloader，模型以及优化器，还有一些日志的保存路径
        Arguments:
            model -- 待训练的模型
            train_dataset -- 训练数据集
            eval_dataset -- 验证数据集
            test_dataset -- 测试数据集
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.train_dataset.coll_fn)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.optim = optim.Adam(self.model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)
        self.epochs = args.epochs
        self.loss_fn = torch.nn.CrossEntropyLoss()

        filename = '+'.join(args.use_list)

        if args.use_base:
            save_file_name='best_base.pt'
        else:
            save_file_name='best_bin.pt'

        if args.main:
            save_file_name='main_'+save_file_name

        if args.use_archi:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, args.architecture, save_file_name)
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)
        else:
            self.save_dir = os.path.join(
                args.para_save_dir, filename, save_file_name)
            tensorboard_dir = os.path.join(
                args.board_log_dir, filename, args.architecture)

        self.tensorboard_dir = os.path.join(
            tensorboard_dir, f'lr:{args.lr}_wd:{args.weight_decay}')
        self.esp = EarlyStopping(self.model, self.save_dir)

    def _train(self, epoch):
        """_summary_
            训练模型
        Arguments:
            epoch -- 第几次训练
        """
        self.model.train()

        self.total_correct = 0
        self.total_loss = 0

        t_bar = tqdm(total=len(self.train_dataloader), desc=f'第{epoch+1}轮训练')
        for graph, data, row_mask, node_mask, label in self.train_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            graph = graph.to(self.device)
            

            res = self.model(graph, data, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            correct = (torch.argmax(res, dim=-1) == label).sum().item()
            self.total_correct += correct
            self.total_loss += loss.item()*res.shape[0]

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _eval(self, epoch):
        """_summary_
            模型验证
        Arguments:
            epoch -- 第几次验证
        """
        self.model.eval()

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        t_bar = tqdm(total=len(self.eval_dataloader), desc=f'第{epoch+1}轮验证')
        for graph, data, row_mask, node_mask, label in self.eval_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            graph = graph.to(self.device)
            

            res = self.model(graph, data, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

            t_bar.write(
                f'loss:{loss.item():8.4f}|accuracy:{correct/res.shape[0]:8.4f}|correct:{correct:4d}')
            t_bar.update()

    @torch.no_grad()
    def _test(self):
        """_summary_
            模型测试
        """
        self.model.eval()

        self.pre_label = list()
        self.label = list()

        self.total_test_correct = 0
        self.total_test_loss = 0

        for graph, data, row_mask, node_mask, label in tqdm(self.test_dataloader, desc='测试集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            graph = graph.to(self.device)
            

            res = self.model(graph, data, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_test_correct += correct
            self.total_test_loss += loss.item()*res.shape[0]

            self.pre_label.extend(
                torch.nn.functional.softmax(res, dim=-1)[:, 1].tolist())
            self.label.extend(label.tolist())

        self.total_correct = 0
        self.total_loss = 0
        for graph, data, row_mask, node_mask, label in tqdm(self.train_dataloader,desc='训练集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            graph = graph.to(self.device)
            

            res = self.model(graph, data, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_correct += correct

        self.total_eval_correct = 0
        self.total_eval_loss = 0
        for graph, data, row_mask, node_mask, label in tqdm(self.eval_dataloader,desc='验证集分析'):
            data = data.to(self.device)
            label = label.to(self.device)
            row_mask = row_mask.to(self.device)
            node_mask = node_mask.to(self.device)
            graph = graph.to(self.device)
            

            res = self.model(graph, data, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()

            self.total_eval_loss += loss.item()*res.shape[0]
            self.total_eval_correct += correct

        acc = self.total_test_correct/len(self.test_dataset)
        loss = self.total_test_loss/len(self.test_dataset)
        score = roc_auc_score(np.array(self.label), np.array(self.pre_label))

        train_acc=self.total_correct/len(self.train_dataset)
        eval_acc=self.total_eval_correct/len(self.eval_dataset)

        logger.info(
            f'train_acc:{train_acc:8.4f}|eval_acc={eval_acc:8.4f}|test dataset acc:{acc:8.4f}|loss:{loss:8.4f}|auc_roc_score:{score:8.6f}')

    def __call__(self) -> None:
        """_summary_
            整个模型的训练过程，首先是一轮训练和验证，结束后判断是否早停，并对日志进行记录和保存
        """
        self.writer = SummaryWriter(self.tensorboard_dir)
        for epoch in range(self.epochs):
            self._train(epoch=epoch)
            self._eval(epoch=epoch)

            self.esp(self.total_eval_loss/len(self.eval_dataset))
            if self.esp.break_now:
                print('early stopping...')
                break

            self.writer.add_scalar(
                'loss_epoch/train', self.total_loss/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/train', self.total_correct/len(self.train_dataset), global_step=epoch)
            self.writer.add_scalar(
                'loss_epoch/eval', self.total_eval_loss/len(self.eval_dataset), global_step=epoch)
            self.writer.add_scalar(
                'acc_epoch/eval', self.total_eval_correct/len(self.eval_dataset), global_step=epoch)

            file_logger.info(f'{args.architecture if args.use_archi else "all":>10s}|epoch:{epoch:4d}|train_loss:{self.total_loss/len(self.train_dataset):8.4f}|train_acc:{self.total_correct/len(self.train_dataset):8.4f}|eval_loss:{self.total_eval_loss/len(self.eval_dataset):8.4f}|eval_acc:{self.total_eval_correct/len(self.eval_dataset):8.4f}')

        self.writer.close()
        self.model.load_state_dict(torch.load(self.save_dir))
        self._test()