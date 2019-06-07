from enum import Enum
from torch.autograd import Variable
from torch import Tensor

class DataPolicy(Enum):
    ALL_DATA = 1


class DataHandler(object):
    def __init__(self, predictors, response, policy: Enum = DataPolicy.ALL_DATA, model_parameters=None, response_shape=(-1, 1)):
        pred_tensors = {}
        for (k,v) in predictors.items():
            pred_tensors[k] = Variable(Tensor(v))
        self.predictors = pred_tensors

        # self.response = Variable(Tensor(response)).view(-1, 1)
        self.response = Variable(Tensor(response)).view(response_shape)
        self.policy = policy

        self.model_parameters = {}
        if model_parameters is not None:
            self.model_parameters = model_parameters
