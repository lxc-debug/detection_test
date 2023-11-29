import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
import subprocess
import shlex
from config.log_conf import file_logger


lr_li = ['1e-3', '1e-4', '1e-5', '1e-6', '1e-7']
weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5']




for lr in lr_li:
    for weight_decay in weight_decay_li:
        try:

            subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay} --use_archi --process_data --logdir {"./grid_log/experiment.log"}'),check=True)
        except subprocess.CalledProcessError:
            file_logger.error('something wrong in run.py')