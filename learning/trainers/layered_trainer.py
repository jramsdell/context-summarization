
from typing import *

from learning.trainers.base_trainer import BaseTrainer
from learning.trainers.data_handler import DataHandler


class LayeredLearner(object):
    def __init__(self, data_handlers: List[DataHandler], trainer_class: Type[BaseTrainer], model, loss_function, lr=0.01, weight_decay=0.05):
        self.base_layers = []

        for data_handler in data_handlers:
            layer = Layer(data_handler=data_handler, trainer_class=trainer_class, model=model, loss_function=loss_function, lr=lr, weight_decay=weight_decay)
            layer.train()
            self.base_layers.append(layer)




class Layer(object):
    def __init__(self, data_handler: DataHandler, trainer_class: Type[BaseTrainer], model, loss_function, lr=0.001, weight_decay=0.05):
        self.trainer = trainer_class(data_handler=data_handler, model=model, loss_function=loss_function, lr=lr)

    def train(self):
        self.trainer.train(n=2000, report=False, weight_decay=0.00)



