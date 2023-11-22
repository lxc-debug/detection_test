import torch
import numpy as np
from config.conf import args
import os


class EarlyStopping():
    def __init__(self, model, save_dir) -> None:
        self.model = model
        self.count = 0
        self.patience = args.patience
        self.loss_now = np.inf
        self.break_now = False
        self.save_dir = save_dir
        self.delta = args.delta

    def __call__(self, loss):
        if loss < self.loss_now-self.delta:
            self.loss_now = loss
            self.count = 0
            self.save_model()

        else:
            self.count += 1
            if self.count > self.patience:
                self.break_now = True

    def save_model(self):
        if not os.path.exists(os.path.dirname(self.save_dir)):
            os.makedirs(os.path.dirname(self.save_dir))
        torch.save(self.model.state_dict(), self.save_dir)
