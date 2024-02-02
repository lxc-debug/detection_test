from model.simple_model import SimpleModel, SimpleModelQ, SimpleModelAttach
from dataset import MyDataset, TestDataset, PosEmbTestDataset, MainDataset
from train import Experiment, ExperimentTest, ExperimentPosEmbTest, MainExperiment
from config.log_conf import logger, file_logger
import torch
import random
import numpy as np
from utils.tar import *
from glob import glob
from model.attention_model import ModelTestNodeAggregate, ModelTestNodePosEmbAggregate, MainModel
from config.conf import args
import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')


# use test
# tar_one()
# print(glob('./log/*'))

my_seed = 0
np.random.seed(seed=my_seed)
random.seed(a=my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

if __name__ == '__main__':

    if not args.main:
        if not args.test:
            file_logger.info(
                f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

            train_dataset = MyDataset(mode='train')
            eval_dataset = MyDataset(mode='eval')
            test_dataset = MyDataset(mode='test')

            if args.use_q:
                model = SimpleModelQ()
            elif args.use_three:
                model = SimpleModelAttach()
            else:
                model = SimpleModel()

            start_train = Experiment(model, train_dataset,
                                    eval_dataset, test_dataset)
            start_train()

        else:
            if not args.add_par_pos_emb:
                file_logger.info(
                    f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

                train_dataset = TestDataset(mode='train')
                eval_dataset = TestDataset(mode='eval')
                test_dataset = TestDataset(mode='test')

                model = ModelTestNodeAggregate()

                start_train = ExperimentTest(
                    model, train_dataset, eval_dataset, test_dataset)
                start_train()

            else:
                file_logger.info(
                    f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

                train_dataset = PosEmbTestDataset(mode='train')
                eval_dataset = PosEmbTestDataset(mode='eval')
                test_dataset = PosEmbTestDataset(mode='test')

                model = ModelTestNodePosEmbAggregate()

                start_train = ExperimentPosEmbTest(
                    model, train_dataset, eval_dataset, test_dataset)
                start_train()

    else:
        file_logger.info(
                    f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

        train_dataset = MainDataset(mode='train')
        eval_dataset = MainDataset(mode='eval')
        test_dataset = MainDataset(mode='test')

        model = MainModel()

        start_train = MainExperiment(
            model, train_dataset, eval_dataset, test_dataset)
        start_train()
