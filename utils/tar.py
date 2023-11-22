import tarfile
import glob
import pandas as pd
from tqdm import tqdm
from torchvision import models
import torch
from config.conf import args
import os


def tar_one():
    res_dir = os.path.join(args.data_path, 'leader_one')

    inv = models.inception_v3(num_classes=5, aux_logits=False)

    train_list = glob.glob('/data/liuxuchao/leaderboard/one/train/*')
    train_df = pd.read_csv(
        '/data/liuxuchao/leaderboard/one/train.csv', delimiter=',', usecols=[0, 1, 10, 15])

    for train_tar in tqdm(train_list, desc='训练数据'):
        with tarfile.open(train_tar, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.split('/')[-1] == 'model.pt':
                    member.name = member.name.split('/')[0]+'.pt'

                    item = train_df.loc[train_df.loc[:]['model_name'].isin(
                        [member.name.split('.')[0]])]
                    is_poisoned = item.loc[:]['ground_truth'].item()
                    archi = item.loc[:]['model_architecture'].item()
                    target = item.loc[:]['trigger_target_class'].item()
                    member.name = (
                        f'poisoned_{archi}_{target}_' if is_poisoned else f'clean_{archi}_')+member.name

                    if is_poisoned:
                        path = os.path.join(
                            res_dir, 'poisoned_models_trainval')

                    else:
                        path = os.path.join(res_dir, 'clean_models_trainval')

                    tar.extract(member, path)
                    if item.loc[:]['model_architecture'].item() == 'inceptionv3':
                        par_name = path+'/'+member.name
                        inv.load_state_dict(torch.load(par_name).state_dict())
                        torch.save(inv, par_name)

    eval_list = glob.glob('/data/liuxuchao/leaderboard/one/eval/*')
    eval_df = pd.read_csv('/data/liuxuchao/leaderboard/one/eval.csv',
                          delimiter=',', usecols=[0, 1, 10, 15])
    for eval_tar in tqdm(eval_list, desc='验证数据'):
        with tarfile.open(eval_tar, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.split('/')[-1] == 'model.pt':
                    member.name = member.name.split('/')[0]+'.pt'

                    item = eval_df.loc[eval_df.loc[:]['model_name'].isin(
                        [member.name.split('.')[0]])]
                    is_poisoned = item.loc[:]['ground_truth'].item()
                    archi = item.loc[:]['model_architecture'].item()
                    target = item.loc[:]['trigger_target_class'].item()
                    member.name = (
                        f'poisoned_{archi}_{target}_' if is_poisoned else f'clean_{archi}_')+member.name

                    if is_poisoned:
                        path = os.path.join(res_dir, 'poisoned_models_eval')
                    else:
                        path = os.path.join(res_dir, 'clean_models_eval')

                    tar.extract(member, path)
                    if item.loc[:]['model_architecture'].item() == 'inceptionv3':
                        par_name = path+'/'+member.name
                        inv.load_state_dict(torch.load(par_name).state_dict())
                        torch.save(inv, par_name)

    test_list = glob.glob('/data/liuxuchao/leaderboard/one/test/*')
    test_df = pd.read_csv('/data/liuxuchao/leaderboard/one/test.csv',
                          delimiter=',', usecols=[0, 1, 10, 15])
    for test_tar in tqdm(test_list, desc='测试数据'):
        with tarfile.open(test_tar, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.split('/')[-1] == 'model.pt':
                    member.name = member.name.split('/')[0]+'.pt'

                    item = test_df.loc[test_df.loc[:]['model_name'].isin(
                        [member.name.split('.')[0]])]
                    is_poisoned = item.loc[:]['ground_truth'].item()
                    archi = item.loc[:]['model_architecture'].item()
                    target = item.loc[:]['trigger_target_class'].item()
                    member.name = (
                        f'poisoned_{archi}_{target}_' if is_poisoned else f'clean_{archi}_')+member.name

                    if is_poisoned:
                        path = os.path.join(res_dir, 'poisoned_models_test')
                    else:
                        path = os.path.join(res_dir, 'clean_models_test')

                    tar.extract(member, path)
                    if item.loc[:]['model_architecture'].item() == 'inceptionv3':
                        try:
                            par_name = path+'/'+member.name
                            inv.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(inv, par_name)
                        except:
                            print(member.name)
                            continue
