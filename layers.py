from activation import Linear, ReLU, Sigmoid, Tanh
import numpy as np
import abc

# attributes base class -> API
class Layer(abc.ABC):
    @abc.abstractmethod
    def build(self, input_dim):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass


class Dense(Layer):
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.linear = None
        self.is_builded = False

    # construct Linear layer first
    def build(self, input_dim):
        self.linear = Linear(input_dim, self.units)
        self.is_builded = True

    def forward(self, x):
        assert self.is_builded
        self.ybar = self.linear.forward(x)
        if self.activation is not None:
            self.ybar = self.activation.forward(self.ybar)
        return self.ybar

    def backward(self, dy):
        assert self.is_builded
        if self.activation is not None:
            dy = self.activation.backward(dy)
        dy = self.linear.backward(dy)
        return dy
