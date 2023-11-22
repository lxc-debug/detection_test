import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
from config.conf import logger, args
from train import Experiment
from model.simple_model import SimpleModel
from dataset import MyDataset
from glob import glob
from utils.tar import *


# use test
# tar_one()
# print(glob('./log/*'))

if __name__ == '__main__':
    train_dataset = MyDataset(mode='train')
    eval_dataset = MyDataset(mode='eval')
    test_dataset = MyDataset(mode='test')

    model = SimpleModel()

    start_train = Experiment(model, train_dataset, eval_dataset, test_dataset)
    start_train()
