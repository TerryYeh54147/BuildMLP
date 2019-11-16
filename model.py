import numpy as np
from activation import Linear, ReLU, Sigmoid, Tanh
from layers import Dense
from loss import MSE
from optimizers import SGD

class Sequential:
    def __init__(self, layers, input_dim):
        self.input_dim = input_dim
        self.layers = layers
        self.loss = None
        self.ybar = None
        self.optimizers = None

    # add layer into the model
    def add(self, layer):
        self.layers.append(layer)

    # start to build layers of the model
    def compile(self, loss, optimizers):
        self.loss = loss
        self.optimizers = optimizers
        input_dim = self.input_dim
        for layer in self.layers:
            layer.build(input_dim)
            input_dim = layer.units

    def forward(self, x):
        self.ybar = x
        for layer in self.layers:
            self.ybar = layer.forward(self.ybar)
        return self.ybar

    def backward(self, y):
        self.loss_val =  self.loss.forward(y, self.ybar)
        dy = self.loss.backward(1)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return self.loss_val

    def fit(self, train_x=None, train_y=None, batch_size=32, epoches=1000, verbose_epoch=100):
        ybar = np.empty(train_y.shape)
        loss_vals = np.empty(epoches)
        # implement batch_size control
        for epoch in range(epoches):
            e_begin = batch_size * epoch
            e_end = e_begin + batch_size
            # take the number of training data and training label that match the batch size cyclically
            slice_x = train_x.take(range(e_begin, e_end),mode='wrap', axis=0)
            slice_y = train_y.take(range(e_begin,e_end),mode='wrap', axis=0)
            slice_ybar = self.forward(slice_x)
            # put results of prediction into the numpy array cyclically
            ybar.put(range(e_begin,e_end), slice_ybar, mode='wrap')
            loss_vals[epoch]= self.backward(slice_y)
            # update weights and biases
            for layer in self.layers:
                layer.linear.w = self.optimizers.update(layer.linear.w,layer.linear.dw)
                layer.linear.b = self.optimizers.update(layer.linear.b,layer.linear.db)
            # print loss per each verbose epoch
            if (epoch+1)%verbose_epoch == 0:
                print('Epoch %3d: loss: %.9f'%(epoch+1, loss_vals[epoch]))
        return ybar, loss_vals
