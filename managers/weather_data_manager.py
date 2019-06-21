import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from learning.models.logistic_model import LogisticModel
from learning.models.logistic_nn_model import LogisticNetModel
from learning.trainers.simple_trainer import SimpleTrainer
from parsing.weather_parser import WeatherParser


class WeatherDataManager(object):
    labels: np.ndarray
    features: np.ndarray

    def __init__(self, weather_loc, trainer, model):
        torch.manual_seed(19238)
        self.wp = WeatherParser(weather_loc)
        self.labels = None
        self.features = None
        self.trainer = trainer
        self.model = model

    def organize_data(self):
        self.features, self.labels = self.wp.parse()
        self.create_train_test_split()


    def create_train_test_split(self):
        # self.features = self.features.T[0:self.features.shape[1] - 10].T
        print(self.features.shape)
        self.x_train, self.x_test, \
        self.y_train, self.y_test = train_test_split(self.features, self.labels,
                                                     test_size = 0.33, random_state = 42)


        self.x_train = self.scale(self.x_train)
        self.x_test = self.scale(self.x_test)
        self.y_train = Tensor(self.y_train)
        # self.y_test = Tensor(self.y_test)
        # self.scale(self.y_train)
        # self.scale(self.y_test)

    def scale(self, x):
        scale = StandardScaler().fit(x)
        return Tensor(scale.transform(x))


    def initialize_trainer(self, lr=0.00001):
        self.trainer = self.trainer(self.x_train, self.y_train.view(-1, 1), self.model, lr)

    def train(self, n_epochs=200):
        self.trainer.train(n_epochs)


    def test(self):
        print("Testing")
        result = (self.trainer.predict(self.x_test)).detach().numpy()
        predictions = np.where(result > 0.5, 1.0, 0.0)

        accuracy = (predictions.squeeze() == self.y_test).sum() / self.y_test.shape[0]

        print("Accuracy: {}".format(accuracy))

        # print((np.where(result.numpy() >= 0.5, 1, 0) == self.y_test).sum() / self.y_test.shape[0])
        # print((result.numpy() == self.y_test).sum())


        # predicted_labels = []
        # for x in self.x_test:
        #     result = (self.trainer.predict(x))
        #     if float(result) >= 0.5:
        #         predicted_labels.append(1.0)
        #     else:
        #         predicted_labels.append(0.0)
        #
        # accuracy = ((self.y_test == np.asarray(predicted_labels))).sum() / self.y_test.shape[0]
        # print("Accuracy: {}".format(accuracy))
        # print(list(self.trainer.model.parameters()))



    @staticmethod
    def load_hardcoded_data():
        trainer = SimpleTrainer
        model = LogisticModel
        wd = WeatherDataManager("/home/jsc57/jupyter/data/weather/weatherAUS.csv", trainer, model)
        wd.organize_data()
        # wd.initialize_trainer()
        # wd.train()
        # wd.test()
        return wd


    @staticmethod
    def do_test():
        trainer = SimpleTrainer
        model = LogisticNetModel
        # model = RNNModel
        # model = LogisticModel
        wd = WeatherDataManager("/home/jsc57/jupyter/data/weather/weatherAUS.csv", trainer, model)
        wd.organize_data()
        wd.initialize_trainer(0.5)
        wd.train()
        wd.test()

if __name__ == '__main__':
    WeatherDataManager.do_test()


