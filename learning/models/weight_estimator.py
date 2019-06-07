import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from learning.trainers.data_handler import DataHandler


class WeightEstimatorModel(torch.nn.Module):
    def __init__(self, data_handler: DataHandler):
        super(WeightEstimatorModel, self).__init__()
        self.n_features = data_handler.predictors["p1"].shape[0]
        self.linear = torch.nn.Linear(self.n_features, 1)
        self.sigmoid = torch.nn.Sigmoid()
        n_weights = data_handler.model_parameters["n_weights"]

        self.p1_transform = torch.nn.Linear(self.n_features, self.n_features)
        self.p2_transform = torch.nn.Linear(self.n_features, self.n_features)
        self.c_transform = torch.nn.Linear(self.n_features, self.n_features)
        self.c_transform2 = torch.nn.Linear(self.n_features, self.n_features)
        self.weight_transform = torch.nn.Linear(self.n_features, n_weights)
        self.weight_transform2 = torch.nn.Linear(self.n_features, n_weights)
        self.weight_transform3 = torch.nn.Linear(self.n_features, n_weights)
        self.weight_to_weight = torch.nn.Linear(n_weights, n_weights)
        self.tanh = nn.Hardtanh()


    def forward(self, data_handler: DataHandler):

        p1 = data_handler.predictors["p1"]
        c = data_handler.predictors["c"]
        p2 =  data_handler.predictors["squished_samples"]

        # p2 = data_handler.predictors["p2"]


        p1 = self.tanh(self.p1_transform(p1.t()))
        p2 = self.tanh(self.p2_transform(p2))
        c = self.tanh(self.c_transform(c.t()))

        # y_pred = self.weight_transform(p1 + p2)
        y_pred = self.weight_transform(p1) + self.weight_transform2(p2) + self.weight_transform3(c)
        # y_pred = self.weight_to_weight(y_pred)
        return y_pred
