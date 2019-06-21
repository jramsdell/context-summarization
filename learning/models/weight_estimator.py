import torch
import torch.nn as nn

from learning.trainers.data_handler import DataHandler


class WeightEstimatorModel(torch.nn.Module):
    def __init__(self, data_handler: DataHandler):
        super(WeightEstimatorModel, self).__init__()
        torch.manual_seed(10)
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

        self.n_layers = 4
        self.wee_weighter = torch.nn.Linear(self.n_layers, 1)
        self.lstm = torch.nn.RNN(self.n_features, self.n_features, bidirectional=True)

        self.ss = torch.nn.Linear(self.n_features, self.n_features * self.n_layers)
        # self.ss_compressor = torch.nn.Linear(self.n_features, self.n_features)
        self.ss_compressor = torch.nn.Linear(self.n_features * 2, self.n_features)
        self.dropout = torch.nn.Dropout(p=0.25, inplace=False)
        self.is_training = True


    def forward(self, data_handler: DataHandler):

        p1 = data_handler.predictors["p1"]
        c = data_handler.predictors["c"]

        p2 =  data_handler.predictors["squished_samples"]

        # p2 = data_handler.predictors["p2"]

        def throughput(x):

            x = x.t()
            embedding = self.ss(x).reshape(self.n_layers, x.shape[0], x.shape[1])
            if self.is_training:
                embedding = self.dropout(embedding)
            wee, _ = self.lstm(embedding)
            if self.is_training:
                wee = self.dropout(wee)

            wee_weighted = self.wee_weighter(wee.permute(1, 2, 0)).squeeze()
            # wee_weighted = wee.mean(0)
            # compressed = self.tanh(self.ss_compressor(wee_weighted))
            compressed = (self.ss_compressor(wee_weighted))
            # compressed = wee_weighted
            return compressed

        # wt = self.weight_transform(throughput(p1) * throughput(c) * throughput(p2.t()))
        wt = self.weight_transform(throughput(c) * throughput(p1))
        return wt




        # p1 = self.tanh(self.p1_transform(p1.t()))
        # p2 = self.tanh(self.p2_transform(p2))
        # c = self.tanh(self.c_transform(c.t()))
        #
        # # y_pred = self.weight_transform(p1) * self.weight_transform2(p2) * self.weight_transform3(c)
        # y_pred = self.weight_transform(p1) * self.weight_transform2(p2) * self.weight_transform3(c)
        # y_pred = self.weight_to_weight(y_pred)
        # y_pred = self.weight_transform(p1.t()) + self.weight_transform2(c.t())
        # return y_pred
