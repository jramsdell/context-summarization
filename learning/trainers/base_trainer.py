from learning.trainers.data_handler import DataHandler
import torch


class BaseTrainer(object):
    def __init__(self, data_handler: DataHandler, model, loss_function, lr=0.001):
        self.data_handler = data_handler
        self.model = model(data_handler)
        self.lr = lr
        self.loss_function = loss_function



    def train(self, n=3000, report=True, weight_decay=0.005):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.step_size = n / 10
        self.counter = 0


        def closure():
            self.optimizer.zero_grad()
            y_pred = self.model(self.data_handler)
            loss = self.loss_function(y_pred, self.data_handler.response)
            self.counter += 1

            if report and self.counter % self.step_size == 0:
                print(loss)
            loss.backward()
            return loss

        for epoch in range(n):
            self.model.train()
            self.optimizer.step(closure)
