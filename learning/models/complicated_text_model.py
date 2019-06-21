import torch
import torch.nn as nn

from learning.trainers.data_handler import DataHandler


class ComplicatedTextModel(torch.nn.Module):
    def __init__(self, data_handler: DataHandler):
        super(ComplicatedTextModel, self).__init__()
        # self.n_features = 40
        self.n_features = data_handler.predictors["p1"].shape[1]

        # self.n_features = n_features


        self.lstm = torch.nn.RNN(self.n_features, self.n_features, bidirectional=False)
        self.n_layers = 2
        # self.ss = [torch.nn.Linear(self.n_features, self.n_features) for i in range(3)]
        self.ss = torch.nn.Linear(self.n_features, self.n_features * self.n_layers)
        # self.ss_compressor = torch.nn.Linear(self.n_features * 2, self.n_features)
        self.ss_compressor = torch.nn.Linear(self.n_features, self.n_features)
        self.p2_trans = (torch.nn.Linear(self.n_features, self.n_features))

        self.wee = torch.nn.Linear(self.n_features, 1)
        self.wee_weighter = torch.nn.Linear(self.n_layers, 1)

        self.tanh = torch.nn.Hardtanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.01, inplace=False)
        self.softtmax = nn.Softmax(dim=0)
        self.is_training = True


    # def forward(self, p1, c, p2):
    def forward(self, data_handler: DataHandler):
        p1 = data_handler.predictors["p1"]
        c = data_handler.predictors["c"]
        p2 = data_handler.predictors["p2"]
        # if self.is_training:
        #     p2 = self.dropout(p2)

        # embedding = torch.vstack([i(p1) for i in self.ss])

        embedding = self.ss(p1).reshape(self.n_layers, p1.shape[0], p1.shape[1])

        # embedding2 = self.ss(p2).reshape(self.n_layers, p1.shape[0], p1.shape[1])

        if self.is_training:
            embedding = self.dropout(self.tanh(embedding))
            # embedding2 = self.dropout(self.tanh(embedding2))





        wee, _ = self.lstm(embedding)
        # wee2, _ = self.lstm(embedding2)

        if self.is_training:
            wee = self.dropout(wee)
            # wee2 = self.dropout(wee2)

        wee_weighted = self.wee_weighter(wee.permute(1, 2, 0)).squeeze()
        # wee_weighted2 = self.wee_weighter(wee2.permute(1, 2, 0)).squeeze()


        compressed = self.tanh(self.ss_compressor(wee_weighted))
        # compressed2 = self.tanh(self.ss_compressor(wee_weighted2))

        if self.is_training:
            compressed = self.dropout(compressed)
            # compressed2 = self.dropout(compressed2)




        # return self.sigmoid((self.tanh(self.p2_trans(p2)) * compressed).sum(1)).unsqueeze(1)
        # return self.sigmoid((self.tanh(compressed) * self.tanh(compressed2)).sum(1)).unsqueeze(1)
        return self.sigmoid((compressed * p2).sum(1)).unsqueeze(1)
        # return self.sigmoid(self.p2_trans(p2).sum(1)).unsqueeze(1)
        # return self.sigmoid((self.tanh(self.p2_trans(p2)) * self.tanh(self.p2_trans(p1))).sum(1)).unsqueeze(1)
        # return self.sigmoid(self.wee(p2))
