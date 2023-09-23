
from pathlib import Path
import requests
from utils import mutation_utils
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def main(model_name):
    DATA_PATH = Path('data')
    PATH = (DATA_PATH / 'mnist')
    PATH.mkdir(parents=True, exist_ok=True)
    URL = 'https://github.com/pytorch/tutorials/raw/main/_static/'
    FILENAME = 'mnist.pkl.gz'
    if (not (PATH / FILENAME).exists()):
        content = requests.get((URL + FILENAME)).content
        (PATH / FILENAME).open('wb').write(content)
    with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    (x_train, y_train, x_valid, y_valid) = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    x_train: 'pytorch_training_data'
    mutation_utils.save_annotated_params(pytorch_training_data=x_train)
    (n, c) = x_train.shape
    lr = 0.5
    lr: 'learning_rate'
    mutation_utils.save_annotated_params(learning_rate=lr)
    epochs = 2
    epochs: 'epochs'
    mutation_utils.save_annotated_params(epochs=epochs)
    bs = 64
    bs: 'batch_size'
    mutation_utils.save_annotated_params(batch_size=bs)
    loss_func = nn.NLLLoss()
    loss_func: 'loss_function'
    mutation_utils.save_annotated_params(loss_function=loss_func)
    optim_func = optim.ASGD
    optim_func: 'optim_algorithm'
    mutation_utils.save_annotated_params(optim_algorithm=optim_func)
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    def get_data(train_ds, valid_ds, batch_size):
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(valid_ds, batch_size=(batch_size * 2)))

    class Mnist_Logistic(nn.Module):

        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(784, 10)
            self.logsoftmax = nn.LogSoftmax()

        def forward(self, xb):
            xb = self.logsoftmax(self.lin1(xb))
            return xb

    def get_model():
        model = Mnist_Logistic()
        return (model, optim_func(model.parameters(), lr=lr))
    (model, opt) = get_model()

    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    validation_scores = []

    def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for (xb, yb) in train_dl:
                train_loss = loss_func(model(xb), yb)
                train_accuracy = accuracy(model(xb), yb)
                train_loss.backward()
                opt.step()
                opt.zero_grad()
            model.eval()
            with torch.no_grad():
                test_loss = (sum((loss_func(model(xb), yb) for (xb, yb) in valid_dl)) / len(valid_dl))
                test_accuracy = (sum((accuracy(model(xb), yb) for (xb, yb) in valid_dl)) / len(valid_dl))
        validation_scores.append(test_loss.item())
        validation_scores.append(test_accuracy.item())
        return validation_scores
    (train_dl, valid_dl) = get_data(train_ds, valid_ds, bs)
    return fit(epochs, model, loss_func, opt, train_dl, valid_dl)
if (__name__ == '__main__'):
    main('')
