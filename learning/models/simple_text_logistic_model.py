import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from learning.trainers.data_handler import DataHandler


class SimpleTextLogisticModel(torch.nn.Module):
    def __init__(self, data_handler: DataHandler):
        super(SimpleTextLogisticModel, self).__init__()
        self.n_features = data_handler.predictors["xs"].shape[0]
        self.linear = torch.nn.Linear(self.n_features, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, data_handler: DataHandler):
        xs = data_handler.predictors["xs"]
        y_pred = self.sigmoid(self.linear(xs.t()))
        return y_pred
