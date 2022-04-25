
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
dataloader
"""



class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, num_feature, stride=1):
        #총 데이터의 개수
        L = y.shape[0]
        # seq_len
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output : shape = (window 크기, sample 개수)
        X = np.zeros([input_window, num_samples, num_feature])
        Y = np.zeros([output_window, num_samples, num_feature])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i,:] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], num_feature).transpose((1,0,2)) # (seq_len, input_window, feature)
        Y = Y.reshape(Y.shape[0], Y.shape[1], num_feature).transpose((1,0,2)) # (seq_len, output_window, feature)
        self.x = X
        self.y = Y
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len


"""
model
"""

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1): # num_layers: lstm layer 수
        super(lstm, self).__init__() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True 인 경우 (batch, input_window, feature) 에 맞춰서 input
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True) 
        self.linear = nn.Linear(self.hidden_size*2, 1)

    def forward(self, x):
        output, hidden = self.lstm(x)
        out = self.linear(output)
        return out, hidden

"""
train
"""

def train(model, train_loader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0
    
    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device).float()
        y = y.to(device).float()
        output, hidden = model(x)
        loss = criterion(output, y[:,:,0].unsqueeze(2))
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
    train_loss = total_loss/len(train_loader)

    return output.detach().cpu().numpy(), y.detach().cpu().numpy(), train_loss

"""
predict
"""


def predict(model, test_loader, criterion, device, file_name=None):
    model.eval()

    total_loss = 0.0
    outputs = []
    ys = []
    for t, (x, y) in enumerate(test_loader):
        # print(t)
        x = x.to(device).float()
        y = y.to(device).float()
        output, hidden = model(x)
        outputs.append(list(output[:,:,0].detach().cpu().numpy()))
        ys.append(list(y[:,:,0].detach().cpu().numpy()))
        loss = criterion(output[:,:,0], y[:,:,0])
        total_loss += loss.cpu().item()
    test_loss = total_loss/len(test_loader)

    return np.array(sum(outputs,[])), np.array(sum(ys,[])), test_loss

