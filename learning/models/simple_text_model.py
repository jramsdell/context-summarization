import torch
import torch.nn as nn

from learning.trainers.data_handler import DataHandler


class SimpleTextModel(torch.nn.Module):
    def __init__(self, data_handler: DataHandler):
        super(SimpleTextModel, self).__init__()
        # self.n_features = 40
        self.n_features = data_handler.predictors["p1"].shape[1]

        # self.n_features = n_features

        self.p1_transform = torch.nn.Linear(self.n_features, self.n_features)
        self.p2_transform = torch.nn.Linear(self.n_features, self.n_features)
        self.c_transform = torch.nn.Linear(self.n_features, self.n_features)
        self.c_transform2 = torch.nn.Linear(self.n_features, self.n_features)


        self.dot_transform = torch.nn.Linear(self.n_features, 1)
        self.tanh = torch.nn.Hardtanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.02, inplace=False)
        self.softtmax = nn.Softmax(dim=1)


    # def forward(self, p1, c, p2):
    def forward(self, data_handler: DataHandler):
        p1 = data_handler.predictors["p1"]
        c = data_handler.predictors["c"]
        p2 = data_handler.predictors["p2"]

        cd = self.sigmoid(self.c_transform(c))
        cd2 = self.sigmoid(self.c_transform2(c))

        # p1d = (self.tanh(self.p1_transform(p1 * cd)))
        p1d = (self.tanh(self.p1_transform(p1 )))
        # p2d = (self.tanh(self.p2_transform(p2 * cd2)))
        # p2d = (self.tanh(self.p2_transform(p2 * cd2)))
        p2d = (self.tanh(self.p2_transform(p2 )))
        # cd = (self.tanh(self.c_transform(c)))
        # p1d = (self.tanh(self.dropout(self.p1_transform(p1))))
        # p2d = (self.tanh(self.dropout(self.p2_transform(p2))))
        # p2d = p2
        # y_pred = self.dot_transform(p1d * p2d * cd)
        # y_pred = self.sigmoid(y_pred)
        # y_pred = (p1d * p2d * cd).sum(1)
        y_pred = (p1d * p2d * cd).sum(1)
        # y_pred = (p1d * p2d * cd + (p1d * p2d* cd2) ).sum(1)

        y_pred = self.sigmoid(y_pred)
        return y_pred.unsqueeze(1)
