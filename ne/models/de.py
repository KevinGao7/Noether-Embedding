from easydict import EasyDict
from torch import nn
from torch.nn import functional as F
from typing import Tuple

import datetime
import torch
import numpy as np

from .base import BaselineModel


class DE_SimplE(BaselineModel):
    def __init__(self, cfg: EasyDict):
        super(DE_SimplE, self).__init__()
        self.set_config(cfg)

        self.ent_embs_h = nn.Embedding(self.sizes[0], self.rank_s)
        self.ent_embs_t = nn.Embedding(self.sizes[0], self.rank_s)
        self.rel_embs_f = nn.Embedding(self.sizes[1], self.rank_s+self.rank_t)
        self.rel_embs_i = nn.Embedding(self.sizes[1], self.rank_s+self.rank_t)
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
        
        self.time_list = [i for i in range(365)]
        fir_day = datetime.datetime(self.year,1,1)
        self.list_m, self.list_d = [], []
        for i in range(len(self.time_list)):
            zone = datetime.timedelta(days=self.time_list[i])
            dat=datetime.datetime.strftime(fir_day + zone, "%Y-%m-%d")
            self.list_m.append(float(dat[5:7]))
            self.list_d.append(float(dat[8:]))
        self.tensor_y = self.year*torch.ones(365)
        self.tensor_m = torch.Tensor(self.list_m)
        self.tensor_d = torch.Tensor(self.list_d)

    def set_config(self, cfg):
        self.sizes = cfg.sizes
        self.rank = cfg.RANK
        self.rank_s = int(cfg.RANK * cfg.SE_PROP)
        self.rank_t = self.rank - self.rank_s
        self.dropout = cfg.DROPOUT
        self.neg_ratio = cfg.NEG_RATIO
        self.year = cfg.YEAR

    def get_device(self):
        return self.ent_embs_h.weight.device
    
    def create_time_embedds(self):
        device = self.get_device()
        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.m_freq_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_freq_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_freq_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_freq_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_freq_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.m_phi_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_phi_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_phi_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_phi_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_phi_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.m_amps_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_amps_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_amps_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_amps_h = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_amps_t = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        device = self.get_device()
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)
        if h_or_t == "head":
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
        else:
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))        
        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        
        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2
    
    def expand(self, x, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(x.cpu()), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.sizes[0], size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.sizes[0], size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0
        
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.sizes[0]
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.sizes[0]
        nnp = np.concatenate((facts1, facts2), axis=0)
        return torch.LongTensor(nnp).cuda()

    def forward(self, x):
        batch_size, time_len = x.shape[0], self.sizes[3]
        times = torch.arange(0, time_len, device=x.device, dtype=torch.long).repeat(x.shape[0]).unsqueeze(1)
        x = x.repeat_interleave(time_len, 0)
        x = torch.cat((x, times), 1)
        score = self.forward_(x, neg_sample=False).view(batch_size, time_len)
        score = self.norm_event_score(score)
        return score
    
    def forward_(self, x, neg_sample=True):
        device = self.get_device()
        if neg_sample:
            x1 = self.expand(x, self.neg_ratio)
        else:
            x1 = x
        tt_y = self.year*torch.ones(x1.size(0)).to(device)
        tt_m = torch.gather(self.tensor_m.to(device), dim=0, index=x1[:, 3])
        tt_d = torch.gather(self.tensor_d.to(device), dim=0, index=x1[:, 3])
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(x1[:, 0], x1[:, 1], x1[:, 2], tt_y, tt_m, tt_d)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores


class DE_DistMult(BaselineModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], 
            rank: int, se_prop: float,
            dropout: float, neg_ratio: int,
            year: int
    ):
        super(DE_DistMult, self).__init__()
        self.sizes = sizes
        self.rank_s = int(rank * se_prop)
        self.rank_t = rank - self.rank_s
        self.dropout = dropout
        self.neg_ratio = neg_ratio
        self.ent_embs = nn.Embedding(self.sizes[0], self.rank_s)
        self.rel_embs = nn.Embedding(self.sizes[1], self.rank_s+self.rank_t)
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        self.year = year
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
        self.time_list = [i for i in range(365)]
        fir_day = datetime.datetime(self.year,1,1)
        self.list_m, self.list_d = [], []
        for i in range(len(self.time_list)):
            zone = datetime.timedelta(days=self.time_list[i])
            dat=datetime.datetime.strftime(fir_day + zone, "%Y-%m-%d")
            self.list_m.append(float(dat[5:7]))
            self.list_d.append(float(dat[8:]))
        self.tensor_y = self.year*torch.ones(365)
        self.tensor_m = torch.Tensor(self.list_m)
        self.tensor_d = torch.Tensor(self.list_d)

    def get_device(self):
        return self.ent_embs.weight.device
    
    def create_time_embedds(self):
        device = self.get_device()
        # frequency embeddings for the entities
        self.m_freq = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_freq = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_freq = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        # phi embeddings for the entities
        self.m_phi = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_phi = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_phi = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        # frequency embeddings for the entities
        self.m_amps = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_amps = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_amps = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        nn.init.xavier_uniform_(self.m_amps.weight)
        nn.init.xavier_uniform_(self.d_amps.weight)
        nn.init.xavier_uniform_(self.y_amps.weight)

    def get_time_embedd(self, entities, years, months, days):
        device = self.get_device()
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)
        emb  = self.y_amps(entities) * self.time_nl(self.y_freq(entities) * years  + self.y_phi(entities))
        emb += self.m_amps(entities) * self.time_nl(self.m_freq(entities) * months + self.m_phi(entities))
        emb += self.d_amps(entities) * self.time_nl(self.d_freq(entities) * days   + self.d_phi(entities))
        
        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        h_embs = self.ent_embs(heads)
        r_embs = self.rel_embs(rels)
        t_embs = self.ent_embs(tails)
        
        h_embs = torch.cat((h_embs, self.get_time_embedd(heads, years, months, days)), 1)
        t_embs = torch.cat((t_embs, self.get_time_embedd(tails, years, months, days)), 1)
        
        return h_embs, r_embs, t_embs
    
    def expand(self, x, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(x.cpu()), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.sizes[0], size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.sizes[0], size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0
        
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.sizes[0]
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.sizes[0]
        nnp = np.concatenate((facts1, facts2), axis=0)
        return torch.LongTensor(nnp).cuda()

    def forward(self, x):  
        s = 0
        for xx in x:
            xxx = xx.expand(365,3)
            h_embs, r_embs, t_embs = self.getEmbeddings(xxx[:, 0], xxx[:, 1], xxx[:, 2], self.tensor_y, self.tensor_m, self.tensor_d)
            scores = (h_embs * r_embs) * t_embs
            scores = F.dropout(scores, p=self.dropout, training=self.training)
            scores = torch.sum(scores, dim=1).unsqueeze(0)
            if s == 0:
                score = scores
            else:
                score = torch.cat([score, scores], dim=0)
            s = 1
        return score
    
    def forward_(self, x):
        device = self.get_device()
        x1 = self.expand(x, self.neg_ratio)
        tt_y = self.year*torch.ones(x1.size(0)).to(device)
        tt_m = torch.gather(self.tensor_m.to(device), dim=0, index=x1[:, 3])
        tt_d = torch.gather(self.tensor_d.to(device), dim=0, index=x1[:, 3])
        h_embs, r_embs, t_embs = self.getEmbeddings(x1[:, 0], x1[:, 1], x1[:, 2], tt_y, tt_m, tt_d)
        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores


class DE_TransE(BaselineModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], 
            rank: int, se_prop: float,
            dropout: float, neg_ratio: int,
            year: int
    ):
        super(DE_TransE, self).__init__()
        self.sizes = sizes
        self.rank_s = int(rank * se_prop)
        self.rank_t = rank - self.rank_s
        self.dropout = dropout
        self.neg_ratio = neg_ratio
        self.ent_embs = nn.Embedding(self.sizes[0], self.rank_s)
        self.rel_embs = nn.Embedding(self.sizes[1], self.rank_s+self.rank_t)
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        self.year = year
        
        self.time_list = [i for i in range(365)]
        fir_day = datetime.datetime(self.year,1,1)
        self.list_m, self.list_d = [], []
        for i in range(len(self.time_list)):
            zone = datetime.timedelta(days=self.time_list[i])
            dat=datetime.datetime.strftime(fir_day + zone, "%Y-%m-%d")
            self.list_m.append(float(dat[5:7]))
            self.list_d.append(float(dat[8:]))
        self.tensor_y = self.year*torch.ones(365)
        self.tensor_m = torch.Tensor(self.list_m)
        self.tensor_d = torch.Tensor(self.list_d)
    
    @staticmethod
    def has_time():
        return True

    def get_device(self):
        return self.ent_embs.weight.device
    
    def create_time_embedds(self):
        device = self.get_device()
        # frequency embeddings for the entities
        self.m_freq = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_freq = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_freq = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        # phi embeddings for the entities
        self.m_phi = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_phi = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_phi = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        # frequency embeddings for the entities
        self.m_amps = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.d_amps = nn.Embedding(self.sizes[0], self.rank_t).to(device)
        self.y_amps = nn.Embedding(self.sizes[0], self.rank_t).to(device)

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        nn.init.xavier_uniform_(self.m_amps.weight)
        nn.init.xavier_uniform_(self.d_amps.weight)
        nn.init.xavier_uniform_(self.y_amps.weight)

    def get_time_embedd(self, entities, years, months, days):
        device = self.get_device()
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)
        emb  = self.y_amps(entities) * self.time_nl(self.y_freq(entities) * years  + self.y_phi(entities))
        emb += self.m_amps(entities) * self.time_nl(self.m_freq(entities) * months + self.m_phi(entities))
        emb += self.d_amps(entities) * self.time_nl(self.d_freq(entities) * days   + self.d_phi(entities))
        
        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        h_embs = self.ent_embs(heads)
        r_embs = self.rel_embs(rels)
        t_embs = self.ent_embs(tails)
        
        h_embs = torch.cat((h_embs, self.get_time_embedd(heads, years, months, days)), 1)
        t_embs = torch.cat((t_embs, self.get_time_embedd(tails, years, months, days)), 1)
        
        return h_embs, r_embs, t_embs
    
    def expand(self, x, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(x.cpu()), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.sizes[0], size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.sizes[0], size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0
        
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.sizes[0]
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.sizes[0]
        nnp = np.concatenate((facts1, facts2), axis=0)
        return torch.LongTensor(nnp).cuda()

    def forward(self, x):  
        s = 0
        for xx in x:
            xxx = xx.expand(365,3)
            h_embs, r_embs, t_embs = self.getEmbeddings(xxx[:, 0], xxx[:, 1], xxx[:, 2], self.tensor_y, self.tensor_m, self.tensor_d)
            scores = h_embs + r_embs - t_embs
            scores = F.dropout(scores, p=self.dropout, training=self.training)
            scores = -torch.norm(scores, dim=1).unsqueeze(0)
            if s == 0:
                score = scores
            else:
                score = torch.cat([score, scores], dim=0)
            s = 1
        return score
    
    def forward_(self, x):
        device = self.get_device()
        x1 = self.expand(x, self.neg_ratio)
        tt_y = self.year*torch.ones(x1.size(0)).to(device)
        tt_m = torch.gather(self.tensor_m.to(device), dim=0, index=x1[:, 3])
        tt_d = torch.gather(self.tensor_d.to(device), dim=0, index=x1[:, 3])
        h_embs, r_embs, t_embs = self.getEmbeddings(x1[:, 0], x1[:, 1], x1[:, 2], tt_y, tt_m, tt_d)
        scores = h_embs + r_embs - t_embs
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores
