import sys
import logging
import argparse
from config.conf import args
from logging.config import dictConfig


log_config = {
    'version': 1,
    'loggers': {
        'root': {
            'level': 'DEBUG',
            'handlers': ['consoleHandler'],
        },
        'mylog': {
            'level': 'DEBUG',
            'handlers': ['fileHandler', 'consoleHandler'],
            'qualname': 'mylog',
            'propagate': 0,
        },
        'filelog':{
            'level':'DEBUG',
            'handlers':['fileHandler'],
            'qualname':'filelog',
            'propagate':0
        },
    },
    'handlers': {
        'consoleHandler': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
        },
        'fileHandler': {
            'class': 'logging.FileHandler',
            'filename': args.logdir,
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
        },
    },
    'formatters': {
        'simpleFormatter': {
            'format': '%(asctime)s|%(levelname)8s|%(filename)s:%(lineno)s|%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
}

# config logger
dictConfig(log_config)

# instance
logger = logging.getLogger('mylog')
file_logger = logging.getLogger('filelog')