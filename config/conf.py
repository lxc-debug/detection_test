import logging
from logging.config import dictConfig
import argparse


# conf paser
# path
parser = argparse.ArgumentParser()
parser.add_argument('--onnx_model', default='./onnx_model',
                    help='position of onnx model')
parser.add_argument('--data_path', default='./data', help='position of data')
parser.add_argument('--data_save', default='./data_save',
                    help='position of processed data')
parser.add_argument('--leaderone_train_data_dir', default=['data/leader_one/poisoned_models_trainval',
                    'data/leader_one/clean_models_trainval'], nargs='+', help='position of leaderone train data')
parser.add_argument('--leaderone_eval_data_dir', default=['data/leader_one/poisoned_models_eval',
                    'data/leader_one/clean_models_eval'], nargs='+', help='position of leaderone eval data')
parser.add_argument('--leaderone_test_data_dir', default=['data/leader_one/poisoned_models_test',
                    'data/leader_one/clean_models_test'], nargs='+', help='position of leaderone test data')
parser.add_argument('--leadertwo_train_data_dir', default=['data/leader_two/poisoned_models_trainval',
                    'data/leader_two/clean_models_trainval'], nargs='+', help='position of leadertwo train data')
parser.add_argument('--leadertwo_eval_data_dir', default=['data/leader_two/poisoned_models_eval',
                    'data/leader_two/clean_models_eval'], nargs='+', help='position of leadertwo eval data')
parser.add_argument('--leadertwo_test_data_dir', default=['data/leader_two/poisoned_models_test',
                    'data/leader_two/clean_models_test'], nargs='+', help='position of leadertwo test data')
parser.add_argument('--vgg_train_data_dir', default=['data/vgg/poisoned_models_trainval',
                    'data/vgg/clean_models_trainval'], nargs='+', help='position of leadertwo train data')
parser.add_argument('--vgg_eval_data_dir', default=['data/vgg/poisoned_models_eval',
                    'data/vgg/clean_models_eval'], nargs='+', help='position of leadertwo eval data')
parser.add_argument('--vgg_test_data_dir', default=['data/vgg/poisoned_models_test',
                    'data/vgg/clean_models_test'], nargs='+', help='position of leadertwo test data')
parser.add_argument('--resnet_train_data_dir', default=['data/resnet/poisoned_models_trainval',
                    'data/resnet/clean_models_trainval'], nargs='+', help='position of leadertwo train data')
parser.add_argument('--resnet_eval_data_dir', default=['data/resnet/poisoned_models_eval',
                    'data/resnet/clean_models_eval'], nargs='+', help='position of leadertwo eval data')
parser.add_argument('--resnet_test_data_dir', default=['data/resnet/poisoned_models_test',
                    'data/resnet/clean_models_test'], nargs='+', help='position of leadertwo test data')
parser.add_argument('--para_save_dir', default='./best_parameter',
                    help='directory for saving best parameter')
parser.add_argument(
    '--board_log_dir', default='./log/tensorboard', help='tensorboard logging dir')
parser.add_argument('--logdir', default='./log/experiment.log',
                    help='directory of log file')

# option
parser.add_argument(
    '--use_list', default=['leader_one'], nargs='+', help='which dataset to use')
parser.add_argument('--architecture', default='resnet50',
                    help='which architecture to select')
parser.add_argument('--embedding_type', default='trans',
                    choices=['trans', 'nerf'], help='how to embedding position')


# switch
parser.add_argument('--use_archi', default=False,
                    action='store_true', help='whether to use architecture')
parser.add_argument('--load_parameter', default=False,
                    action='store_true', help='whether to load model to onnx')
parser.add_argument('--process_data', default=False,
                    action='store_true', help='whether to process the raw data')
parser.add_argument('--use_base', default=False, action='store_true',
                    help='whether use base method to process data')
parser.add_argument('--use_three', default=False,
                    action='store_true', help='whether add architecture data')
parser.add_argument('--use_q',default=False,action='store_true',help='whether use q to aggregate row_size')
parser.add_argument('--test',default=False,action='store_true',help='whether test')
parser.add_argument('--use_q_node',default=False,action='store_true',help='whether use q to aggregate node_size')
parser.add_argument('--add_par_pos_emb',default=False,action='store_true',help='whether add position embedding to parameter')
parser.add_argument('--main',default=False,action='store_true',help='whether implement main experiment')


# hyperparameters
parser.add_argument('--bin_num', default=14, type=int, help='number of bins')
parser.add_argument('--hidden_dim', default=32, type=int,
                    help='attention hidden dimemsion')
parser.add_argument('--patience', default=100, type=int,
                    help='patience of early stopping')
parser.add_argument('--delta', default=0, type=float,
                    help='delta for early stopping')
parser.add_argument('--batch_size', default=20, type=int,
                    help='batch size for train eval test')
parser.add_argument('--epochs', default=1000000,
                    type=int, help='epochs for train')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate for train')
parser.add_argument('--weight_decay', default=0,
                    type=float, help='weight_decay for train')
parser.add_argument('--padding_dim', default=30,
                    type=int, help='padding dim for data')
parser.add_argument('--row_size', default=2048, type=int,
                    help='row size of a feature')
parser.add_argument('--op_type_size', default=215,
                    type=int, help='size of operator type')
parser.add_argument('--pos_emb_dim', default=32, type=int,
                    help='the dimension of position embedding')
# 得有这个参数来保证row_size这个列表是一个等大的，这样比较好操作
parser.add_argument('--num_nodes', default=400, type=int,
                    help='the number of all graph nodes')
parser.add_argument('--n_head',default=8,type=int,help='number of the attention head')


# op_dir
with open('op_list.txt', mode='r', encoding='utf16') as fp:
    content = fp.read()
    op_li = content.split(',')

parser.add_argument('--op_li', default=op_li, type=list,
                    help='list of operate type')


# instance
args = parser.parse_args()
