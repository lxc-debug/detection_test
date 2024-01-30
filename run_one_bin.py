import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
import subprocess
import shlex
from config.log_conf import file_logger


# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/all_noarchi_base_seed1.log"}  --use_list {"leader_one"} --use_base'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')

# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/all_noarchi_bin_seed1.log"}  --use_list {"leader_one"}'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')


# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/all_archi_base_useq_seed1.log"}  --use_list {"leader_one"} --use_base --use_three --use_q'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')


# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/all_archi_bin_useq_seed1.log"}  --use_list {"leader_one"} --use_three --use_q'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')


# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/all_archi_base_attach.log"}  --use_list {"leader_one"} --use_base --use_three'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')


# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/all_archi_bin_attach.log"}  --use_list {"leader_one"} --use_three'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')

# lr_li = ['1e-5','1e-6']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/save/inception_base_two.log"}  --use_list {"leader_one"}  --use_archi --architecture {"inceptionv3"}'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')

# 新的实验
# lr_li = ['1e-1','1e-2','1e-3', '1e-4']
# weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4','1e-3','1e-2','1']

# for lr in lr_li:
#     for weight_decay in weight_decay_li:
#         try:
#             subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/test_save/base.log"}  --use_list {"leader_one"} --use_base --test'),check=True)
#         except subprocess.CalledProcessError:
#             file_logger.error('something wrong in run.py')


# subprocess.run(shlex.split(f'python main.py --lr {0.1} --weight_decay {0}   --logdir {"./log_tmp/test_save/debug_bin.log"}  --use_list {"leader_one"} --test --process_data'))

lr_li = ['1e-1','1e-2','1e-3', '1e-4']
weight_decay_li = ['0', '1e-5','1e-4','1e-3','1e-2',]

for weight_decay in weight_decay_li:
    for lr in lr_li:
        try:
            subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}   --logdir {"./log_tmp/test_save/bin.log"}  --use_list {"leader_one"} --test'),check=True)
        except subprocess.CalledProcessError:
            file_logger.error('something wrong in run.py')