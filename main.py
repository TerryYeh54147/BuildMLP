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
            # count the number of 1
            cnt += int(trainingData[i][j])
        # give label answer
        labels.append(1 if cnt % 2 else 0)
    return labels

def generate_training_data():
    # generate the training data
    train_data = list(itertools.product([0,1], repeat=8))
    # get the training label
    train_label = get_training_lebal(train_data)
    return np.array(train_data), np.reshape(train_label, (-1,1))

def loss_diagram(epochs,losses):
    plt.plot(losses)
    # give x label name
    plt.xlabel('Epoch')
    # give x label name
    plt.ylabel('Loss')
    # save the loss diagram
    plt.savefig(f'{losses[-1]:0.9f}.png')

def output_results(train_y, predict,last_loss):
    labels, preds = [] , []
    # convert the numpy array to list
    for (l,p) in zip(train_y,predict):
        if l[0]!=int(round(p[0])):
            print(f'{l[0],int(round(p[0]))}')
        labels.append(l[0])
        # round the result of my prediction
        preds.append(int(round(p[0])))
    # build a dictionary for tabulation
    res = {
        'label': labels,
        'predict': preds
    }
    data = pd.DataFrame(res)
    print("————————————————————————————")
    print(f"final loss: {last_loss:.9f}",end="\n\n")
    print(data)
    # save the result by the final loss value
    data.to_csv(f'{last_loss:.9f}.csv')

def main():
    # get training data and training label
    train_x, train_y = generate_training_data()
    # get the shape of training data
    batch_train_x, input_train_x = train_x.shape
    learn_rate = 0.01
    momentum = 0.
    epoches = 200
    verbose_epoch = 10
    # declare the model
    models = Sequential(
        layers = [
        Dense(128, activation=ReLU()),
        Dense(64, activation=Tanh()),
        Dense(28, activation=Sigmoid())],
        input_dim = input_train_x
    )
    # add the layer
    models.add(Dense(16, activation=ReLU()))
    models.add(Dense(1, activation=Sigmoid()))
    # declare the optimizer
    sgd = SGD(learn_rate = learn_rate, momentum = momentum)
    # start to build the model
    models.compile(loss = MSE(), optimizers = sgd)
    # start to training
    predict, losses = models.fit(train_x, train_y, batch_size = batch_train_x, epoches=epoches, verbose_epoch = verbose_epoch)
    # draw the loss diagram
    loss_diagram(epoches,losses)
    # output the results into the csv file
    output_results(train_y,predict,losses[-1])

if __name__ == "__main__":
    # use the random seed to fixed my result
    np.random.seed(10)
    main()
