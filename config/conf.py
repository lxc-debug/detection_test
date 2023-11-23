import logging
from logging.config import fileConfig
import argparse

# conf logging
fileConfig('./config/logging.conf')

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
parser.add_argument('--para_save_dir', default='./best_parameter',
                    help='directory for saving best parameter')
parser.add_argument(
    '--board_log_dir', default='./log/tensorboard', help='tensorboard logging dir')

# option
parser.add_argument(
    '--use_list', default=['leader_one'], nargs='+', help='which dataset to use')
parser.add_argument('--architecture', default='resnet50',
                    help='which architecture to select')

# switch
parser.add_argument('--use_archi', default=False,
                    action='store_true', help='whether to use architecture')
parser.add_argument('--load_parameter', default=False,
                    action='store_true', help='whether to load model to onnx')
parser.add_argument('--process_data', default=False,
                    action='store_true', help='whether to process the raw data')
parser.add_argument('--use_base',default=False,action='store_true',help='whether use base method to process data')

# hyperparameters
parser.add_argument('--bin_num', default=11, type=int, help='number of bins')
parser.add_argument('--hidden_dim', default=32, type=int,
                    help='attention hidden dimemsion')
parser.add_argument('--patience', default=100, type=int,
                    help='patience of early stopping')
parser.add_argument('--delta', default=0, type=float,
                    help='delta for early stopping')
parser.add_argument('--batch_size', default=200, type=int,
                    help='batch size for train eval test')
parser.add_argument('--epochs', default=1000, type=int, help='epochs for train')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate for train')
parser.add_argument('--weight_decay', default=0,
                    type=float, help='weight_decay for train')


# instance
logger = logging.getLogger('mylog')
file_logger = logging.getLogger('filelog')
args = parser.parse_args()
