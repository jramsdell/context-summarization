import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class SimpleTrainer(object):

    def __init__(self, xs, ys, model: torch.nn.Module, lr=0.01):
        xs = Tensor(xs)
        ys = Tensor(ys)
        self.dataset = TensorDataset(xs, ys)
        self.lr = lr

        self.x_data = Variable(xs)
        self.y_data = Variable(ys)
        self.model = model(xs.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.criterion = torch.nn.BCELoss(size_average=True)
        self.criterion = torch.nn.BCELoss(reduction='mean')
        # self.criterion = torch.nn.NLLLoss(reduction='mean')




    def train(self, n = 100):
        # dloader = DataLoader(self.dataset, batch_size=500)
        # rs = RandomSampler(self.dataset, replacement=False, num_samples=1000)
        rs = RandomSampler(self.dataset, replacement=False)
        dloader = DataLoader(batch_size=500, dataset=self.dataset, sampler=rs)
        wee = 0
        # for (i1, i2) in dloader:
        # print(wee)
        # self.optimizer.weight_decay = 0.5

        # for (i1, i2) in range(n):
        for i in range(20):
            for (i1, i2) in dloader:
                # print(i1.shape)
                self.model.train()
                self.optimizer.zero_grad()

                # Forward
                # y_pred = self.model(self.x_data)
                y_pred = self.model(i1)

                # loss = self.criterion(y_pred, self.y_data)
                # loss = self.criterion(y_pred, i2) + torch.abs(self.model.l1.weight).sum() * 0.01
                loss = self.criterion(y_pred, i2)
                loss.backward()
                self.optimizer.step()


    def train_text(self, tvectors, cvectors, n=3000):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        tvectors = Variable(Tensor(tvectors))
        cvectors = Variable(Tensor(cvectors))

        def closure():
            self.optimizer.zero_grad()
            y_pred = self.model(tvectors, cvectors, self.x_data)
            loss = self.criterion(y_pred, self.y_data)
            print(loss)
            loss.backward()
            return loss

        for epoch in range(n):
            self.model.train()

            self.optimizer.step(closure)


    def train_rnn(self, n=100):
        rs = RandomSampler(self.dataset, replacement=False)
        dloader = DataLoader(batch_size=500, dataset=self.dataset, sampler=rs)

        counter = 0

        for i in range(1):
            for (i1, i2) in dloader:
                counter += 1
                i11 = i1.unsqueeze(0)
                wee = torch.cat([i11 * 2, i11 * 5, i11 * -2])
                # wee = torch.cat([i11 * 2, i11 * 5, i11 * -2])

                # i1 = Variable(Tensor([i1 * 2, i2 * 5, i2 * 10]))


                # if counter % 100 == 0:
                #     break
                # i1 = i1.unsqueeze(0)
                self.model.train()
                self.optimizer.zero_grad()
                # output = self.model(wee)
                output = self.model(i1.unsqueeze(1))

                loss = self.criterion(output, i2)
                loss.backward(retain_graph=True)
                self.optimizer.step()



    def train_rnn_old(self, n=100):
        rs = RandomSampler(self.dataset, replacement=False)
        dloader = DataLoader(batch_size=1, dataset=self.dataset, sampler=rs)
        self.model.init_hidden()

        hidden = torch.zeros(1, 100)
        counter = 0



        for i in range(1):
            for (i1, i2) in dloader:
                counter += 1
                if counter % 100 == 0:
                    break
                self.model.train()
                self.optimizer.zero_grad()


                output, next_hidden = self.model(input, hidden)
                hidden = next_hidden

                loss = self.criterion(output, i2)
                loss.backward(retain_graph=True)
                self.optimizer.step()

        self.hidden = hidden


    def predict(self, x):
        return self.model(x)



