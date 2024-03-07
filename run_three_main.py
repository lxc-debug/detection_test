import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
import subprocess
import shlex
from config.log_conf import file_logger
import os




lr_li = ['1e-2','1e-3', '1e-4']
weight_decay_li = ['0', '1e-5','1e-4','1e-3','1e-2',]

for weight_decay in weight_decay_li:
    for lr in lr_li:
        try:
            subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}   --logdir {"./log_tmp/main_archi/base_three.log"}  --use_list {"leader_three"} --main --use_base --use_q_node --process_data'),check=True)
        except subprocess.CalledProcessError:
            file_logger.error('something wrong in run.py')

