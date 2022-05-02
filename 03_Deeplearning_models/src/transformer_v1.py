
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random

from torch.utils.data import DataLoader, Dataset
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple

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
model

attn_forward
- multihead attention layer 에서 attention energy 값 저장하도록 class 수정
- need_weights=True 로 설정

"""


def attn_forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    is_batched = query.dim() == 3
    if self.batch_first and is_batched:
        query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    if not self._qkv_same_embed_dim:
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=True,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
    else:
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=True,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)
    
    # property 추가
    self.attn = attn_output_weights

    if self.batch_first and is_batched:
        return attn_output.transpose(1, 0), attn_output_weights
    else:
        return attn_output, attn_output_weights

class transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128): 
        super(transformer, self).__init__() 
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward

        # batch_first=True 인 경우 (batch, input_window, feature) 에 맞춰서 input
        self.transformer = nn.Transformer(d_model=self.d_model, 
                                          nhead=self.nhead,
                                          num_encoder_layers=self.num_encoder_layers,
                                          num_decoder_layers=self.num_decoder_layers,
                                          dim_feedforward=self.dim_feedforward,
                                          batch_first=True) 
        # self.linear = nn.Linear(self.d_model, 1)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        # output2 = self.linear(output1)
        return output

"""
train
"""

# decoding 을 max len 만큼 반복해야해서 시간이 훨씬 길어짐
def train(model, train_loader, max_len, optimizer, criterion, device='cuda'):
    model.train()
    total_loss = 0.0
    attention_list = []

    for x, y_in, y_out in train_loader:
        optimizer.zero_grad()
        x = x.to(device).float()
        y_in = y_in.to(device).float()
        y_out = y_out.to(device).float()

        y_init = x[:,-1,:].unsqueeze(1)
        y_in = y_init.clone().detach()
        for i in range(max_len):
            output = model(x, y_in).to(device)
            y_in = torch.cat([y_init, output], dim=1) 
        attention_list.append(list(model.transformer.decoder.layers[0].multihead_attn.attn.detach().cpu().numpy()))
        # attention_list.append(list(model.transformer.decoder.layers[0].multihead_attn.attn))

        loss = criterion(output, y_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
    train_loss = total_loss/len(train_loader)

    return output.detach().cpu().numpy(), y_out.detach().cpu().numpy(), np.array(sum(attention_list,[])), train_loss

"""
predict
"""

# def predict(model, test_loader, tgt_mask, criterion, device='cuda', file_name=None):
#     model.eval()

#     total_loss = 0.0
#     outputs = []
#     ys = []
#     for t, (x, y_in, y_out) in enumerate(test_loader):
#         # print(t)
#         x = x.to(device).float()
#         y_in = y_in.to(device).float()
#         y_out = y_out.to(device).float()
#         output = model(x, y_in, tgt_mask=tgt_mask).to(device)
#         outputs.append(list(output.detach().cpu().numpy()))
#         ys.append(list(y_out.detach().cpu().numpy()))
#         loss = criterion(output[:,:,0], y_out[:,:,0])
#         total_loss += loss.cpu().item()
#     test_loss = total_loss/len(test_loader)

#     return np.array(sum(outputs,[])), np.array(sum(ys,[])), test_loss


def predict(model, test_loader, max_len, criterion, device='cuda', file_name=None):
    model.eval()

    total_loss = 0.0
    outputs = []
    ys = []
    attention_list = []

    for t, (x, y_in, y_out) in enumerate(test_loader):
        # print(t)
        x = x.to(device).float()
        y_in = y_in.to(device).float()
        y_out = y_out.to(device).float()

        y_init = x[:,-1,:].unsqueeze(1)
        y_in = y_init.detach()
        for i in range(max_len):
            output = model(x, y_in).to(device)
            y_in = torch.cat([y_init, output], dim=1) 
        attention_list.append(list(model.transformer.decoder.layers[0].multihead_attn.attn.detach().cpu().numpy()))
        outputs.append(list(output.detach().cpu().numpy()))
        ys.append(list(y_out.detach().cpu().numpy()))
        loss = criterion(output, y_out)
        total_loss += loss.cpu().item()
    test_loss = total_loss/len(test_loader)

    return np.array(sum(outputs,[])), np.array(sum(ys,[])), np.array(sum(attention_list, [])), test_loss