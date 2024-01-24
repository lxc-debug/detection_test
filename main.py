import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
from config.conf import args
from config.log_conf import logger,file_logger
from train import Experiment
from model.simple_model import SimpleModel,SimpleModelQ
from dataset import MyDataset
from glob import glob
from utils.tar import *
import numpy as np
import random
import torch


# use test
# tar_one()
# print(glob('./log/*'))

my_seed=1
np.random.seed(seed=my_seed)
random.seed(a=my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

if __name__ == '__main__':

    file_logger.info(f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

    train_dataset = MyDataset(mode='train')
    eval_dataset = MyDataset(mode='eval')
    test_dataset = MyDataset(mode='test')

    if args.use_q:
        model = SimpleModelQ()
    else:
        model = SimpleModel()

    start_train = Experiment(model, train_dataset, eval_dataset, test_dataset)
    start_train()
