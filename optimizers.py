import abc
import numpy as np

class Optimizers(abc.ABC):
    @abc.abstractmethod
    def update(self):
        pass

class SGD(Optimizers):
    def __init__(self, learn_rate, momentum):
        self.v = None
        self.lr = learn_rate
        self.m = momentum

    def update(self, change, gradient):
        self.v = np.zeros(change.shape)
        # velocity = momentum * velocity - learn_rate * gradient  # velocity
        self.v = self.m*self.v - self.lr*gradient
        # update the weight or bias
        change = change + self.v
        return change
