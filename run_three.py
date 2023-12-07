import shlex
import subprocess
import re
import logging

logger=logging.getLogger('run_three')
logger.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler=logging.FileHandler('./log_tmp/run_three_all.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


lr_li = ['1e-1','1e-2','1e-3', '1e-4', '1e-5']
weight_decay_li = ['0', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5','1e-4']

def get_info(file_name):

    with open(file_name, 'r') as f:
        log_info = f.read()

    last_epoch_info = re.findall(r"epoch:.*?\|train_loss:.*?train_acc:\s*([\d.]+)\|.*?eval_acc:\s*([\d.]+)", log_info)
    test_dataset_info = re.findall(r"test dataset acc:\s*([\d.]+)\|.*?auc_roc_score:(\d+\.\d+)", log_info)

    if last_epoch_info:
        last_train_acc, last_eval_acc = last_epoch_info[-1] # find last epoch
    
    if test_dataset_info:
        test_acc,auc_roc_score = test_dataset_info[-1]
    
    return float(last_train_acc), float(last_eval_acc), float(test_acc), float(auc_roc_score)



for lr in lr_li:
    for weight_decay in weight_decay_li:
        res=subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/res_tmp.log"}  --use_list {"leader_one"} --use_archi --architecture {"resnet50"}'),check=True)
        
        res_last_train_acc, res_last_eval_acc, res_test_acc, res_auc_roc_score=get_info("./log_tmp/res_tmp.log")

        res=subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/den_tmp.log"}  --use_list {"leader_one"} --use_archi --architecture {"densenet121"}'),check=True)

        den_last_train_acc, den_last_eval_acc, den_test_acc, den_auc_roc_score=get_info("./log_tmp/den_tmp.log")

        res=subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/inv_tmp.log"}  --use_list {"leader_one"} --use_archi --architecture {"inceptionv3"}'),check=True)

        inv_last_train_acc, inv_last_eval_acc, inv_test_acc, inv_auc_roc_score=get_info("./log_tmp/inv_tmp.log")

        res=subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}  --process_data --logdir {"./log_tmp/all_tmp.log"}  --use_list {"leader_one"} --use_three'),check=True)

        all_last_train_acc, all_last_eval_acc, all_test_acc, all_auc_roc_score=get_info("./log_tmp/all_tmp.log")

        logger.info(f'lr {lr}; weight_decay {weight_decay}')
        logger.info(f'all_last_train_acc {all_last_train_acc}; all_last_eval_acc {all_last_eval_acc}; all_test_acc {all_test_acc}; all_auc_roc_score {all_auc_roc_score}')
        logger.info(f'last_train_acc {(res_last_train_acc+den_last_train_acc+inv_last_train_acc)/3:.4f}; last_eval_acc {(res_last_eval_acc+den_last_eval_acc+inv_last_eval_acc)/3:.4f}; test_acc {(res_test_acc+den_test_acc+inv_test_acc)/3:.4f}; auc_roc_score {(res_auc_roc_score+den_auc_roc_score+inv_auc_roc_score)/3:.4f}')

