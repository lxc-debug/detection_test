import tarfile
import glob
import pandas as pd
from tqdm import tqdm
from torchvision import models
import torch
from config.conf import args
import os
import shutil


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


def tar_two():
    res_dir = os.path.join(args.data_path, 'leader_two')

    train_list = glob.glob('/data/liuxuchao/leaderboard/two/train/*')
    train_df = pd.read_csv('/data/liuxuchao/leaderboard/two/train.csv',
                           delimiter=',', usecols=[0, 1, 9, 15, 26])

    for train_tar in tqdm(train_list, desc='训练数据'):
        with tarfile.open(train_tar, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.split('/')[-1] == 'model.pt':
                    member.name = member.name.split('/')[-2]+'.pt'

                    item = train_df.loc[train_df.loc[:]['model_name'].isin(
                        [member.name.split('.')[0]])]
                    is_poisoned = item.loc[:]['poisoned'].item()
                    archi = item.loc[:]['model_architecture'].item()
                    target = item.loc[:]['trigger_target_class'].item()
                    member.name = (
                        f'poisoned_{archi}_{target}_' if is_poisoned else f'clean_{archi}_')+member.name

                    if is_poisoned:
                        path = os.path.join(
                            res_dir, 'poisoned_models_trainval')

                    else:
                        path = os.path.join(res_dir, 'clean_models_trainval')

                    if item.loc[:]['model_architecture'].item() == 'mobilenetv2':
                        continue  # for this is a test program so I don't process this architecture

                    else:
                        tar.extract(member, path)
                        if item.loc[:]['model_architecture'].item() == 'inceptionv3':
                            inv = models.inception_v3(
                                num_classes=item.loc[:]['number_classes'].item(), aux_logits=False)
                            par_name = os.path.join(path, member.name)
                            inv.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(inv, par_name)

                        elif item.loc[:]['model_architecture'].item().find('vgg') != -1:
                            name = item.loc[:]['model_architecture'].item()
                            if name == 'vgg11bn':
                                vgg = models.vgg11_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg13bn':
                                vgg = models.vgg13_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg16bn':
                                vgg = models.vgg16_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg19bn':
                                vgg = models.vgg19_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            else:
                                raise TypeError('没有对应的vgg模型')
                            par_name = os.path.join(path, member.name)
                            vgg.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(vgg, par_name)

    eval_list = glob.glob('/data/liuxuchao/leaderboard/two/eval/*')
    eval_df = pd.read_csv('/data/liuxuchao/leaderboard/two/eval.csv',
                          delimiter=',', usecols=[0, 1, 9, 15, 26])
    for eval_tar in tqdm(eval_list, desc='验证数据'):
        with tarfile.open(eval_tar, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.split('/')[-1] == 'model.pt':
                    member.name = member.name.split('/')[-2]+'.pt'

                    item = eval_df.loc[eval_df.loc[:]['model_name'].isin(
                        [member.name.split('.')[0]])]
                    is_poisoned = item.loc[:]['poisoned'].item()
                    archi = item.loc[:]['model_architecture'].item()
                    target = item.loc[:]['trigger_target_class'].item()
                    member.name = (
                        f'poisoned_{archi}_{target}_' if is_poisoned else f'clean_{archi}_')+member.name

                    if is_poisoned:
                        path = os.path.join(res_dir, 'poisoned_models_eval')
                    else:
                        path = os.path.join(res_dir, 'clean_models_eval')

                    if item.loc[:]['model_architecture'].item() == 'mobilenetv2':
                        continue

                    else:
                        tar.extract(member, path)
                        if item.loc[:]['model_architecture'].item() == 'inceptionv3':
                            inv = models.inception_v3(
                                num_classes=item.loc[:]['number_classes'].item(), aux_logits=False)
                            par_name = os.path.join(path, member.name)
                            inv.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(inv, par_name)

                        elif item.loc[:]['model_architecture'].item().find('vgg') != -1:
                            name = item.loc[:]['model_architecture'].item()
                            if name == 'vgg11bn':
                                vgg = models.vgg11_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg13bn':
                                vgg = models.vgg13_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg16bn':
                                vgg = models.vgg16_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg19bn':
                                vgg = models.vgg19_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            else:
                                raise TypeError('没有对应的vgg模型')
                            par_name = os.path.join(path, member.name)
                            vgg.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(vgg, par_name)

    test_list = glob.glob('/data/liuxuchao/leaderboard/two/test/*')
    test_df = pd.read_csv('/data/liuxuchao/leaderboard/two/test.csv',
                          delimiter=',', usecols=[0, 1, 9, 15, 26])
    for test_tar in tqdm(test_list, desc='测试数据'):
        with tarfile.open(test_tar, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.split('/')[-1] == 'model.pt':
                    member.name = member.name.split('/')[-2]+'.pt'

                    item = test_df.loc[test_df.loc[:]['model_name'].isin(
                        [member.name.split('.')[0]])]
                    is_poisoned = item.loc[:]['poisoned'].item()
                    archi = item.loc[:]['model_architecture'].item()
                    target = item.loc[:]['trigger_target_class'].item()
                    member.name = (
                        f'poisoned_{archi}_{target}_' if is_poisoned else f'clean_{archi}_')+member.name

                    if is_poisoned:
                        path = os.path.join(res_dir, 'poisoned_models_test')
                    else:
                        path = os.path.join(res_dir, 'clean_models_test')

                    if item.loc[:]['model_architecture'].item() == 'mobilenetv2':
                        continue

                    else:
                        tar.extract(member, path)
                        if item.loc[:]['model_architecture'].item() == 'inceptionv3':
                            inv = models.inception_v3(
                                num_classes=item.loc[:]['number_classes'].item(), aux_logits=False)
                            par_name = os.path.join(path, member.name)
                            inv.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(inv, par_name)

                        elif item.loc[:]['model_architecture'].item().find('vgg') != -1:
                            name = item.loc[:]['model_architecture'].item()
                            if name == 'vgg11bn':
                                vgg = models.vgg11_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg13bn':
                                vgg = models.vgg13_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg16bn':
                                vgg = models.vgg16_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            elif name == 'vgg19bn':
                                vgg = models.vgg19_bn(
                                    num_classes=item.loc[:]['number_classes'].item())
                            else:
                                raise TypeError('没有对应的vgg模型')
                            par_name = os.path.join(path, member.name)
                            vgg.load_state_dict(
                                torch.load(par_name).state_dict())
                            torch.save(vgg, par_name)


def tar_vgg():
    res_dir = os.path.join(args.data_path, 'vgg')

    clean_train_list = glob.glob(
        '/data/liuxuchao/model_data/vgg/clean_models_trainval/*')
    poison_train_list = glob.glob(
        '/data/liuxuchao/model_data/vgg/poisoned_models_trainval/*')

    clean_train_dest = os.path.join(res_dir, 'clean_models_trainval')
    poison_train_dest = os.path.join(res_dir, 'poisoned_models_trainval')
    if not os.path.exists(clean_train_dest):
        os.makedirs(clean_train_dest)
    if not os.path.exists(poison_train_dest):
        os.makedirs(poison_train_dest)

    for item in tqdm(clean_train_list, desc='tar clean train'):
        shutil.copy(item, clean_train_dest)

    for item in tqdm(poison_train_list, desc='tar poison train'):
        shutil.copy(item, poison_train_dest)

    # for only have test data, split test to eval and test data
    clean_li = glob.glob('/data/liuxuchao/model_data/vgg/clean_models_test/*')
    poison_li = glob.glob(
        '/data/liuxuchao/model_data/vgg/poisoned_models_test/*')
    clean_li.sort()
    poison_li.sort()

    clean_eval_list = clean_li[:-30]
    poison_eval_list = poison_li[:30]

    clean_eval_dest = os.path.join(res_dir, 'clean_models_eval')
    poison_eval_dest = os.path.join(res_dir, 'poisoned_models_eval')
    if not os.path.exists(clean_eval_dest):
        os.makedirs(clean_eval_dest)
    if not os.path.exists(poison_eval_dest):
        os.makedirs(poison_eval_dest)

    for item in tqdm(clean_eval_list, desc='tar clean eval'):
        shutil.copy(item, clean_eval_dest)

    for item in tqdm(poison_eval_list, desc='tar poison eval'):
        shutil.copy(item, poison_eval_dest)

    clean_test_list = clean_li[-30:]
    poison_test_list = poison_li[-30:]

    clean_test_dest = os.path.join(res_dir, 'clean_models_test')
    poison_test_dest = os.path.join(res_dir, 'poisoned_models_test')
    if not os.path.exists(clean_test_dest):
        os.makedirs(clean_test_dest)
    if not os.path.exists(poison_test_dest):
        os.makedirs(poison_test_dest)

    for item in tqdm(clean_test_list, desc='tar clean test'):
        shutil.copy(item, clean_test_dest)

    for item in tqdm(poison_test_list, desc='tar poison test'):
        shutil.copy(item, poison_test_dest)


def tar_resnet():
    res_dir = os.path.join(args.data_path, 'resnet')

    clean_train_list = glob.glob(
        '/data/liuxuchao/model_data/tiny_image/clean_models_train/*')
    poison_train_list = glob.glob(
        '/data/liuxuchao/model_data/tiny_image/poisoned_models_train/*')

    clean_train_dest = os.path.join(res_dir, 'clean_models_trainval')
    poison_train_dest = os.path.join(res_dir, 'poisoned_models_trainval')
    if not os.path.exists(clean_train_dest):
        os.makedirs(clean_train_dest)
    if not os.path.exists(poison_train_dest):
        os.makedirs(poison_train_dest)

    for item in tqdm(clean_train_list, desc='tar clean train'):
        shutil.copy(item, clean_train_dest)

    for item in tqdm(poison_train_list, desc='tar poison train'):
        shutil.copy(item, poison_train_dest)

    # for only have test data, split test to eval and test data
    clean_li = glob.glob('/data/liuxuchao/model_data/tiny_image/clean_models_test/*')
    poison_li = glob.glob(
        '/data/liuxuchao/model_data/tiny_image/poisoned_models_test/*')
    clean_li.sort()
    poison_li.sort()

    clean_eval_list = clean_li[:-30]
    poison_eval_list = poison_li[:30]

    clean_eval_dest = os.path.join(res_dir, 'clean_models_eval')
    poison_eval_dest = os.path.join(res_dir, 'poisoned_models_eval')
    if not os.path.exists(clean_eval_dest):
        os.makedirs(clean_eval_dest)
    if not os.path.exists(poison_eval_dest):
        os.makedirs(poison_eval_dest)

    for item in tqdm(clean_eval_list, desc='tar clean eval'):
        shutil.copy(item, clean_eval_dest)

    for item in tqdm(poison_eval_list, desc='tar poison eval'):
        shutil.copy(item, poison_eval_dest)

    clean_test_list = clean_li[-30:]
    poison_test_list = poison_li[-30:]

    clean_test_dest = os.path.join(res_dir, 'clean_models_test')
    poison_test_dest = os.path.join(res_dir, 'poisoned_models_test')
    if not os.path.exists(clean_test_dest):
        os.makedirs(clean_test_dest)
    if not os.path.exists(poison_test_dest):
        os.makedirs(poison_test_dest)

    for item in tqdm(clean_test_list, desc='tar clean test'):
        shutil.copy(item, clean_test_dest)

    for item in tqdm(poison_test_list, desc='tar poison test'):
        shutil.copy(item, poison_test_dest)
