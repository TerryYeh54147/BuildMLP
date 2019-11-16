import numpy as np
import abc

# attributes base class -> API
class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, y, ybar):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass

class MSE(Loss):
    def __init__(self):
        self.y, self.ybar = None, None

    def forward(self, y, ybar):
        self.y = y
        # ybar -> 1Xn : 1Xn(ybar)
        self.ybar = ybar
        return np.mean((y-ybar)**2)

    def backward(self, dout):
        # dy -> 1Xn : 1Xn (dout) * 1Xn(-2*(y-ybar)/n)
        dybar = dout * (-2*(self.y - self.ybar))
        return dybar
