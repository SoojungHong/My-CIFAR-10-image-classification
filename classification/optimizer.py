
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Optimizer():

    def optimizer_fn(model):
        learning_rate = 0.001
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    def adam_optimizer_fn(model):
        learning_rate = 1e-6
        return optim.Adam(model.parameters(), lr=learning_rate)

    def sgd_optimizer_fn(model):
        learning_rate = 0.001
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
