import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class LogisticNetModel(torch.nn.Module):
    def __init__(self, n_dims):
        super(LogisticNetModel,self).__init__()
        self.n_dims = n_dims
        self.l1 = torch.nn.Linear(n_dims, n_dims)
        self.l2 = torch.nn.Linear(n_dims, 1)
        self.relu=torch.nn.Hardtanh()
        # self.relu=torch.nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.01, inplace=False)


    def forward(self,x):
        # out1 = self.relu(self.l1(x))
        out1 = (self.l1(x))
        # out1 = self.dropout(out1)

        y_pred = self.sigmoid(self.l2(self.relu(out1)).sum(1))
        return y_pred.unsqueeze(1)




class OldLogisticNetModel(torch.nn.Module):
    def __init__(self, n_dims):
        super(OldLogisticNetModel,self).__init__()
        self.n_dims = n_dims
        self.l1 = torch.nn.Linear(n_dims, 1000)
        self.l2 = torch.nn.Linear(1000, 500)
        self.l3 = torch.nn.Linear(500, 100)
        # self.l4 = torch.nn.Linear(20, 1)
        self.l4 = torch.nn.Linear(100, 1)
        self.sigmoid=torch.nn.Sigmoid()
        self.relu=torch.nn.Hardtanh()
        # self.tanh = torch.nn.Tanh()
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.0, inplace=False)
    def forward(self,x):
        # cur = self.relu(torch.nn.Linear(self.n_dims, 10)(x))

        # for i in range(4):
        #     drop = self.dropout(cur)
        #     cur = self.relu(torch.nn.Linear(10, 10)(drop))

        # drop = self.dropout(cur)
        # cur = self.relu(torch.nn.Linear(10, 4)(drop))
        #
        # y_pred = self.sigmoid(self.l4(cur))
        # return y_pred


        out1=self.relu(self.l1(x))
        drop1=self.dropout(out1)


        # out2=self.relu(self.l2(drop1))
        out2=self.relu(self.l2(drop1))
        drop2=self.dropout(out2)

        out3=self.relu(self.l3((drop2)))
        drop3=self.dropout(out3)


        y_pred=self.sigmoid(self.l4(drop3))
        # y_pred=self.sigmoid(self.l4(drop3))
        return y_pred

