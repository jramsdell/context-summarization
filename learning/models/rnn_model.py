import torch
import torch.nn as nn
from torch.autograd import Variable




class RNNModel(torch.nn.Module):
    def __init__(self, n_dims):
        super(RNNModel,self).__init__()
        self.hidden_size = 100
        self.n_layers = 2

        # self.i2h = nn.Linear(n_dims + self.hidden_size, self.hidden_size)
        # self.i2o = nn.Linear(n_dims + self.hidden_size, 1)

        self.n_dims = n_dims
        self.sigmoid = torch.nn.Sigmoid()

        self.rnn = nn.RNN(
            input_size=self.n_dims,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            nonlinearity='relu',
            batch_first=True # maybe change this later
        )


            # n_dims, self.hidden_size, self.n_layers, batch_first=True, nonlinearity='relu')
        self.output = nn.Linear(self.hidden_size, 1)



    # def forward(self, input, hidden):
    def forward(self, input):
        # combined = torch.cat((input.t(), hidden.t()), 0).t()
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # output = self.sigmoid(output)

        h0 = Variable(torch.zeros(self.n_layers, input.shape[0], self.hidden_size))
        out, hn = self.rnn(input, h0)
        out = self.sigmoid(self.output(out[:, -1, :]))


        # return output, hidden
        return out


    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

