from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

import torch
import numpy as np

from .base import BaselineModel


class TeRo(BaselineModel):
    def __init__(self, cfg):
        super(TeRo, self).__init__()
        self.set_config(cfg)
        
        # Nets
        self.emb_E_real = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.emb_E_img = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.emb_R_real = torch.nn.Embedding(self.sizes[1]*2, self.rank, padding_idx=0)
        self.emb_R_img = torch.nn.Embedding(self.sizes[1]*2, self.rank, padding_idx=0)
        self.emb_Time = torch.nn.Embedding(self.n_day, self.rank, padding_idx=0)
        
        # Initialization
        r = 6 / np.sqrt(self.rank)
        self.emb_E_real.weight.data.uniform_(-r, r)
        self.emb_E_img.weight.data.uniform_(-r, r)
        self.emb_R_real.weight.data.uniform_(-r, r)
        self.emb_R_img.weight.data.uniform_(-r, r)
        self.emb_Time.weight.data.uniform_(-r, r)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.time_tensor = torch.LongTensor([i for i in range(365)])
    @staticmethod
    def has_time():
        return True

    def set_config(self, cfg: EasyDict):
        self.sizes = cfg.sizes
        self.rank = cfg.RANK
        self.gamma = cfg.GAMMA
        self.n_day = cfg.sizes[3]
        self.neg_ratio = cfg.NEG_RATIO

    def get_device(self):
        return self.emb_E_real.weight.device

    def sample_negatives(self, X):
        X1 = np.copy(X.cpu())
        M = X1.shape[0]
        X_corr = X1       
        for i in range(self.neg_ratio-1):
            X_corr = np.concatenate((X_corr,X1),0)
        X_corr[:int(M*self.neg_ratio/2),0]=torch.randint(self.sizes[0],[int(M*self.neg_ratio/2)])        
        X_corr[int(M*self.neg_ratio/2):,2]=torch.randint(self.sizes[0],[int(M*self.neg_ratio/2)]) 
        return torch.LongTensor(X_corr).cuda()

    def log_rank_loss(self, X, temp=0.5):
        y_pos = self.forward_(X)
        y_neg = self.forward_(self.sample_negatives(X))
        M = y_pos.size(0)
        N = y_neg.size(0)
        y_pos = self.gamma-y_pos
        y_neg = self.gamma-y_neg
        C = int(N / M)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        #print(y_neg.size())
        p = F.softmax(temp * y_neg)
        loss_pos = torch.sum(F.softplus(-1 * y_pos))
        loss_neg = torch.sum(p * F.softplus(y_neg))
        loss = (loss_pos + loss_neg) / 2 / M
        return loss

    def forward(self, x):
        batch_size, time_len = x.shape[0], self.sizes[3]
        times = torch.arange(0, time_len, device=x.device, dtype=torch.long).repeat(x.shape[0]).unsqueeze(1)
        x = x.repeat_interleave(time_len, 0)
        x = torch.cat((x, times), 1)
        score = self.forward_(x).view(batch_size, time_len)
        score = self.norm_event_score(score)
        return score

    def forward_(self, X):
        h_i, t_i, r_i, d_i = X[:, 0], X[:, 2], X[:, 1], X[:, 3]
        pi = torch.pi
        d_img = torch.sin(self.emb_Time(d_i).view(-1, self.rank))
        d_real = torch.cos(self.emb_Time(d_i).view(-1, self.rank))
        h_real = self.emb_E_real(h_i).view(-1, self.rank) *d_real-\
                 self.emb_E_img(h_i).view(-1, self.rank) *d_img
        t_real = self.emb_E_real(t_i).view(-1, self.rank) *d_real-\
                 self.emb_E_img(t_i).view(-1, self.rank)*d_img
        r_real = self.emb_R_real(r_i).view(-1, self.rank)
        h_img = self.emb_E_real(h_i).view(-1, self.rank) *d_img+\
                 self.emb_E_img(h_i).view(-1, self.rank) *d_real
        t_img = self.emb_E_real(t_i).view(-1, self.rank) *d_img+\
                self.emb_E_img(t_i).view(-1, self.rank) *d_real
        r_img = self.emb_R_img(r_i).view(-1, self.rank)
        out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
        out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
        out = out_real + out_img
        return out
