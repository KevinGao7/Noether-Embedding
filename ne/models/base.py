from abc import ABC, abstractmethod
from torch.nn import functional as F
from torch import nn
from scipy import signal

import torch
import numpy as np


class TAModel(nn.Module, ABC):
    @abstractmethod
    def event_score(self, event):
        pass

    @abstractmethod
    def ta_score(self, ta, rule_type='specific'):
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    def ta_query(self, ta, ta_score_mode='direct', time_range=None):
        if ta_score_mode == 'corr':
            # print('corr branch')
            scores = self.ta_score_corr(ta)
        else:
            # print('ne branch')
            scores = self.ta_score(ta)
        
            if ta_score_mode == 'norm':
                scores_exp = np.exp(scores)
                scores = scores_exp / scores_exp.sum(1)[:, np.newaxis]


        if time_range is not None:
            scores = scores[ :, time_range[0] + self.sizes[3] : time_range[1] + self.sizes[3]]
            st = time_range[0] + self.sizes[3]
        else:
            st = 0

        val = np.max(scores, 1)
        pos = np.argmax(scores, 1) + st

        return val, pos
        
    def ta_score_corr(self, ta):
        try:
            tmp = self.sizes[0]
        except:
            self.sizes = self.cfg.sizes
        ta = self.to_batch_tensor(ta)
        x1 = torch.index_select(ta, dim=1, index=torch.tensor([0, 1, 2], device=self.get_device()))
        x2 = torch.index_select(ta, dim=1, index=torch.tensor([3, 4, 5], device=self.get_device()))
        score1 = self.event_score(x1)[:, : self.sizes[3]]
        score2 = self.event_score(x2)[:, : self.sizes[3]]

        assert not torch.isnan(score1).any()
        assert not torch.isnan(score2).any()

        def calc_norm_sum(vec):
            return np.sum(vec * vec, axis=1, keepdims=True)

        def calc_norm_score(vec, reverse=False):
            mul = vec * vec
            rev = np.flip(mul, axis=1)
            lef = np.cumsum(mul, axis=1)
            rig = np.flip(np.cumsum(rev[:, 1: ], axis=1), axis=1)
            res = np.concatenate((lef, rig), axis=1)
            if reverse:
                res = np.flip(res, axis=1)
            return res

        scores = np.zeros((ta.shape[0], self.sizes[3] * 2 - 1))
        # sum1 = calc_norm_score(score1.cpu().numpy(), reverse=True)
        # sum2 = calc_norm_score(score2.cpu().numpy(), reverse=False)
        for i in range(scores.shape[0]):
            scores[i, : ] = signal.correlate(score2[i].cpu().numpy(), score1[i].cpu().numpy(), 'full')
        # overlap_lens = np.concatenate((np.arange(1, self.sizes[3]), np.flip(np.arange(1, self.sizes[3] + 1), axis=[0])))
        # scores = scores / overlap_lens
        # scores = scores / np.sqrt(sum1 + 1e-9) / np.sqrt(sum2 + 1e-9)
        scores = scores / np.sqrt(calc_norm_sum(score1.cpu().numpy()) + 1e-9) / np.sqrt(calc_norm_sum(score2.cpu().numpy()) + 1e-9)
        return scores

    def get_device(self):
        try:
            return list(self.modules())[-1].weight.device
        except:
            return self.device

    def to_batch_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.LongTensor(data)
        if len(data.shape) < 2:
            data = data.unsqueeze(0)
        if data.device != self.get_device():
            data = data.to(self.get_device())
        return data


class BaselineModel(TAModel):
    def event_score(self, event):
        with torch.no_grad():
            x = self.to_batch_tensor(event)
            score = self.forward(x)
        return score

    def ta_score(self, ta, rule_type='specific'):
        return self.ta_score_corr(ta)

    def norm_event_score(self, x):
        l, r = x.min(1, keepdim=True)[0], x.max(1, keepdim=True)[0]
        x =  (x - l) / (r - l + 1e-9)
        return x