import torch
import torch.nn as nn

from learning.trainers.data_handler import DataHandler


class BertLSTMModel(torch.nn.Module):
    def __init__(self, data_handler: DataHandler):
        super(BertLSTMModel, self).__init__()
        # self.n_features = 40
        self.n_features = data_handler.predictors["p1"].shape[2]
        n_layers = int(self.n_features / 40)
        # n_layers = int(4)

        # self.linear = torch.nn.Linear(self.n_features, 1)
        self.linear1 = torch.nn.Linear(n_layers, 1)
        self.is_train = True
        self.linear2 = torch.nn.Linear(4, 1)
        self.linear3 = torch.nn.Linear(n_layers, 1)

        # self.lstm_p1 = nn.LSTM(
        #     input_size=self.n_features,
        #     num_layers=4,
        #     hidden_size=4,
        #     batch_first=True
        # )

        self.lstm_p1 = nn.LSTM(
            self.n_features,
            n_layers,
            batch_first=True,
            bidirectional=False
        )

        self.lstm_p2 = nn.LSTM(
            self.n_features,
            n_layers,
            batch_first=True,
            bidirectional=False
        )

        self.dot_transform = torch.nn.Linear(self.n_features, 1)
        self.linear_p1 = torch.nn.Linear(self.n_features, self.n_features)
        self.linear_p2 = torch.nn.Linear(self.n_features, self.n_features)
        self.tanh = torch.nn.Hardtanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.0, inplace=False)
        self.softtmax = nn.Softmax(dim=1)


    # def forward(self, p1, c, p2):
    def forward(self, data_handler: DataHandler):
        p1 = data_handler.predictors["p1"]
        p2 = data_handler.predictors["p2"]

        #
        # p1 = self.tanh(self.linear_p1(p1.mean(1)))
        # p2 = self.tanh(self.linear_p2(p2.mean(1)))

        # if self.is_train:
        #     p1 = p1 * (1.0 + torch.FloatTensor(*p1.shape).uniform_(0.0, 0.05))
        #     p2 = p2 * (1.0 + torch.FloatTensor(*p2.shape).uniform_(0.0, 0.05))


        # return self.sigmoid(self.dot_transform(p1 * p2))
        # return (self.dot_transform(p1 * p2))


        wee, _ = self.lstm_p1(p1)
        wee2, _ = self.lstm_p1(p2)

        if self.is_train:
            wee = self.dropout(wee)
            wee2 = self.dropout(wee2)

        wee = wee.sum(1)
        wee2 = wee2.sum(1)

        y_pred = self.linear3((wee) + (wee2))
        return y_pred

