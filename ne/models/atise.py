from easydict import EasyDict
from torch.nn import functional as F

import torch
import numpy as np

from .base import BaselineModel


class ATISE(BaselineModel):
    def __init__(self, cfg: EasyDict):
        super(ATISE, self).__init__()
        
        self.set_config(cfg)
        
        # Nets
        self.emb_E = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.emb_E_var = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.emb_R = torch.nn.Embedding(self.sizes[1], self.rank, padding_idx=0)
        self.emb_R_var = torch.nn.Embedding(self.sizes[1], self.rank, padding_idx=0)
        self.emb_TE = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.alpha_E = torch.nn.Embedding(self.sizes[0], 1, padding_idx=0)
        self.beta_E = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.omega_E = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.emb_TR = torch.nn.Embedding(self.sizes[1], self.rank, padding_idx=0)
        self.alpha_R = torch.nn.Embedding(self.sizes[1], 1, padding_idx=0)
        self.beta_R = torch.nn.Embedding(self.sizes[1], self.rank, padding_idx=0)
        self.omega_R = torch.nn.Embedding(self.sizes[1], self.rank, padding_idx=0)

        # Initialization
        r = 6 / np.sqrt(self.rank)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.uniform_(-r, r)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.uniform_(-r, r)
        self.alpha_E.weight.data.uniform_(0, 0)
        self.beta_E.weight.data.uniform_(0, 0)
        self.omega_E.weight.data.uniform_(-r, r)
        self.emb_TR.weight.data.uniform_(-r, r)
        self.alpha_R.weight.data.uniform_(0, 0)
        self.beta_R.weight.data.uniform_(0, 0)
        self.omega_R.weight.data.uniform_(-r, r)

        self.time_tensor = torch.LongTensor([i for i in range(365)])
    @staticmethod
    def has_time():
        return True

    def set_config(self, cfg: EasyDict):
        self.rank = cfg.RANK
        self.gamma = cfg.GAMMA
        self.n_day = cfg.N_DAY
        self.neg_ratio = cfg.NEG_RATIO
        self.cmin = cfg.CMIN
        self.cmax = 100 * cfg.CMIN
        self.sizes = cfg.sizes

    def get_device(self):
        return self.emb_E.weight.device

    def regularization_embeddings(self):
        device = self.get_device()
        lower = torch.tensor(self.cmin).float().to(device)
        upper = torch.tensor(self.cmax).float().to(device)
        self.emb_E_var.weight.data=torch.where(self.emb_E_var.weight.data < self.cmin,lower,self.emb_E_var.weight.data)
        self.emb_E_var.weight.data=torch.where(self.emb_E_var.weight.data > self.cmax,upper,self.emb_E_var.weight.data)
        self.emb_R_var.weight.data=torch.where(self.emb_R_var.weight.data < self.cmin,lower, self.emb_R_var.weight.data)
        self.emb_R_var.weight.data=torch.where(self.emb_R_var.weight.data > self.cmax,upper, self.emb_R_var.weight.data)
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)
    
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
        pi = 3.14159265358979323846
        h_mean = self.emb_E(h_i).view(-1, self.rank) + \
            d_i.view(-1, 1) * self.alpha_E(h_i).view(-1, 1) * self.emb_TE(h_i).view(-1, self.rank) \
            + self.beta_E(h_i).view(-1, self.rank) * torch.sin(2 * pi * self.omega_E(h_i).view(-1, self.rank) * d_i.view(-1, 1))
        t_mean = self.emb_E(t_i).view(-1, self.rank) + \
            d_i.view(-1, 1) * self.alpha_E(t_i).view(-1, 1) * self.emb_TE(t_i).view(-1, self.rank) \
            + self.beta_E(t_i).view(-1, self.rank) * torch.sin(2 * pi * self.omega_E(t_i).view(-1, self.rank) * d_i.view(-1, 1))  
        r_mean = self.emb_R(r_i).view(-1, self.rank) + \
            d_i.view(-1, 1) * self.alpha_R(r_i).view(-1, 1) * self.emb_TR(r_i).view(-1, self.rank) \
            + self.beta_R(r_i).view(-1, self.rank) * torch.sin(2 * pi * self.omega_R(r_i).view(-1, self.rank) * d_i.view(-1, 1))
        h_var = self.emb_E_var(h_i).view(-1, self.rank)
        t_var = self.emb_E_var(t_i).view(-1, self.rank)
        r_var = self.emb_R_var(r_i).view(-1, self.rank)
        out1 = torch.sum((h_var+t_var)/r_var, 1)+torch.sum(((r_mean-h_mean+t_mean)**2)/r_var, 1)-self.rank
        out2 = torch.sum(r_var/(h_var+t_var), 1)+torch.sum(((h_mean-t_mean-r_mean)**2)/(h_var+t_var), 1)-self.rank
        out = (out1+out2)/4
        return out
