import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from model import Sequential
from layers import Dense
from activation import Linear, ReLU, Sigmoid, Tanh
from loss import MSE
from optimizers import SGD

def get_training_lebal(trainingData):
    labels = []
    for i in range(len(trainingData)):
        cnt = 0
        for j in range(len(trainingData[i])):
            cnt += int(trainingData[i][j])
        labels.append(1 if cnt % 2 else 0)
    return labels

def generate_training_data():
    # generate training data
    train_data = list(itertools.product([0,1], repeat=8))
    train_label = get_training_lebal(train_data)
    return np.array(train_data), np.reshape(train_label, (-1,1))

def loss_diagram(epochs,losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

def output_results(train_y, predict,last_loss):
    labels, preds = [] , []
    for (l,p) in zip(train_y,predict):
        labels.append(l[0])
        preds.append(int(round(p[0])))
    res = {
        'label': labels,
        'predict': preds
    }
    data = pd.DataFrame(res)
    print("--------------------------------")
    print(f"final loss: {last_loss:.9f}",end="\n\n")
    print(data)

def main():
    train_x, train_y = generate_training_data()
    batch_train_x, input_train_x = train_x.shape
    learn_rate = 0.01
    momentum = 0.
    epoches = 200
    verbose_epoch = 10
    print("learn_rate: {}\nmomentum: {}\n".format(learn_rate, momentum))
    models = Sequential(
        layers = [
        Dense(128, activation=ReLU()),
        Dense(64, activation=Tanh()),
        Dense(28, activation=Sigmoid())],
        input_dim = input_train_x
    )
    models.add(Dense(16, activation=ReLU()))
    models.add(Dense(1, activation=Sigmoid()))
    sgd = SGD(learn_rate = learn_rate, momentum = momentum)
    models.compile(loss = MSE(), optimizers = sgd)
    predict, losses = models.fit(train_x, train_y, batch_size = batch_train_x, epoches=epoches, verbose_epoch = verbose_epoch)

    loss_diagram(epoches,losses)
    output_results(train_y,predict,losses[-1])

if __name__ == "__main__":
    np.random.seed(10)
    main()
