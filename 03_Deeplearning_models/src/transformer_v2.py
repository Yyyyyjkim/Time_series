
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
        Y_src = np.zeros([output_window, num_samples, num_feature])
        Y_tgt = np.zeros([output_window, num_samples, num_feature])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i,:] = y[start_x:end_x]

            start_y = stride*i + input_window 
            end_y = start_y + output_window 
            Y_src[:,i] = y[(start_y-1):(end_y-1)]
            Y_tgt[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], num_feature).transpose((1,0,2)) # (seq_len, input_window, feature)
        Y_src = Y_src.reshape(Y_src.shape[0], Y_src.shape[1], num_feature).transpose((1,0,2)) # (seq_len, output_window, feature)
        Y_tgt = Y_tgt.reshape(Y_tgt.shape[0], Y_tgt.shape[1], num_feature).transpose((1,0,2))
        self.x = X[:,:-1,:]
        self.y_src = Y_src
        self.y_tgt = Y_tgt
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y_src[i], self.y_tgt[i]

    def __len__(self):
        return self.len


"""
train
"""

def train(model, train_loader, optimizer, criterion, max_len, device='cuda'):
    model.train()
    total_loss = 0.0

    for x, y_in, y_out in train_loader:
        optimizer.zero_grad()
        x = x.to(device).float()
        y_in = y_in.to(device).float()
        y_out = y_out.to(device).float()

        for t in range(max_len):
            y_in_temp = y_in[:,:(t+1),:]
            output = model(x, y_in_temp, tgt_mask=None).to(device)
            if t < (max_len-1):
                y_in[:,1:(t+2),:] = output[:,:(t+1),:]
        loss = criterion(output[:,:,0], y_out[:,:,0])
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
    train_loss = total_loss/len(train_loader)

    return output.detach().cpu().numpy(), y_out.detach().cpu().numpy(), train_loss


"""
predict
"""

def predict(model, test_loader, criterion, max_len, device='cuda', file_name=None):
    model.eval()

    total_loss = 0.0
    outputs = []
    ys = []
    for x, y_in, y_out in test_loader:
        # print(t)
        x = x.to(device).float()
        y_in = y_in.to(device).float()
        y_out = y_out.to(device).float()

        for t in range(max_len):
            y_in_temp = y_in[:,:(t+1),:]
            output = model(x, y_in_temp, tgt_mask=None).to(device)
            if t < (max_len-1):
                y_in[:,1:(t+2),:] = output[:,:(t+1),:]
        outputs.append(list(output.detach().cpu().numpy()))
        ys.append(list(y_out.detach().cpu().numpy()))
        loss = criterion(output[:,:,0], y_out[:,:,0])
        total_loss += loss.cpu().item()
    test_loss = total_loss/len(test_loader)

    return np.array(sum(outputs,[])), np.array(sum(ys,[])), test_loss


