import numpy as np
import abc

# attributes base class -> API
class Activation(abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass

class Linear(Activation):
    def __init__(self, input_dim, output_dim):
        # W -> mXn(input_dimXoutput_dim), b -> 1Xn
        self.w, self.b = np.random.randn(input_dim, output_dim), np.random.randn(1,output_dim)
        # dW -> mXn, db -> 1Xn
        self.dw, self.db = None, None

    def forward(self, x):
        # x -> kXm
        self.x = x
        # out -> kXn : kXm(x) · mXn(W) + 1Xn(b)
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        # dx -> kXm: kXn(dout) · nXm(W.T)
        dx = np.dot(dout, self.w.T)
        # dw -> mXn : mXk(x.T) · kXn(dout)
        self.dw = np.dot(self.x.T, dout)
        # db -> 1Xn : 1Xn(dout[0])
        self.db = np.reshape(np.sum(dout, axis=0), (1,-1))
        return dx

class ReLU(Activation):
    def __init__(self):
        self.are_negative = None

    def forward(self, x):
        # areNegative -> kXn : kXn(x;bool(x<0))
        self.are_negative = (x<0)
        # out -> kXn : kXn(x)
        out = x
        # change to zero if it is negative
        out[self.are_negative] = 0
        return out

    def backward(self, dout):
        # dx -> kXn: kxn(dout)
        dx = dout
        # change to zero if it is negative
        dx[self.are_negative] = 0
        return dx

class Sigmoid(Activation):
    def __init__(self):
        self.out = None

    def forward(self, x):
        # out -> kXn : kXn( 1 / (1+exp(-x)) )
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # dx -> kXn : kXn(dout) * kXn(out) * kXn(1-out)
        dx = dout*self.out*(1-self.out)
        return dx

class Tanh(Activation):
    def __init__(self):
        self.out = None

    def forward(self, x):
        # find from website
        # out -> kXn : kXn( (exp(x)-exp(-x)) / (exp(x)-exp(-x)) )
        out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # derivatives of tanh
        # dx -> kXn : kXn( dout * ((1-o)**2) )
        dx = dout*(1-(self.out)**2)
        return dx
