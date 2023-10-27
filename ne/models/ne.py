from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

import math
import torch

from .base import TAModel


class NE(TAModel):
    def __init__(self, cfg: EasyDict, info_dict):
        super(NE, self).__init__()

        self.set_config(cfg, info_dict)

        print('NE.size', self.sizes)

        self.ent_cnt = self.sizes[0]
        self.pre_cnt = self.sizes[1]
        self.time_cnt = self.sizes[3]
        
        self.param_num()

        # T embed
        self.down_omg = math.exp(math.log(2 * math.pi * self.omg_max) / self.rank)
        if self.omg_init == 'linear':
            self.base_omg = torch.arange(0, self.rank) * 2 * math.pi * self.omg_max / self.rank
        else:
            self.base_omg = torch.pow(self.down_omg, torch.arange(0, self.rank)) - 1

        # S, P, O embed
        mod_size = 1 / math.sqrt(self.rank)
        self.register_parameter('tim_omg', torch.nn.Parameter(self.base_omg, requires_grad=False))

        # SPO embed
        if self.embed_mode == 'spo':
            self.add_module('event_type_c', nn.Embedding(len(list(self.embed_dict['spo'].keys())), self.rank * 2))
            self.event_type_c.weight.data *= mod_size
            print('Event type size: ', len(list(self.embed_dict['spo'].keys())))
        elif self.embed_mode == 's,p,o':
            self.add_module('ent_c', nn.Embedding(self.ent_cnt, self.rank * 2))
            self.add_module('pre_c', nn.Embedding(self.pre_cnt, self.rank * 2))
            self.ent_c.weight.data *= mod_size
            self.pre_c.weight.data *= mod_size
        elif self.embed_mode == 'sp,po':
            self.add_module('sp_c', nn.Embedding(len(list(self.embed_dict['sp'].keys())), self.rank * 2))
            self.add_module('po_c', nn.Embedding(len(list(self.embed_dict['po'].keys())), self.rank * 2))
            self.sp_c.weight.data *= mod_size
            self.po_c.weight.data *= mod_size
            print('SP size: ', len(list(self.embed_dict['sp'].keys())))
            print('PO size: ', len(list(self.embed_dict['po'].keys())))

        if self.event_score_mode == 'l2':
            self.add_module('h', nn.Embedding(len(list(self.embed_dict['spo'].keys())), self.rank * 2))
            self.h.weight.data *= mod_size

    def param_num(self):
        if self.embed_mode == 'spo':
            size_evt = (len(list(self.embed_dict['spo'].keys())), self.rank * 2)
            print('param shape', size_evt)
            num = size_evt[0] * size_evt[1]
            print('param num', num)
        elif self.embed_mode == 's,p,o':
            size_ent = self.ent_cnt, self.rank * 2
            size_pre = self.pre_cnt, self.rank * 2
            print('param shape', size_ent, size_pre)
            num = size_ent[0] * size_ent[1] + size_pre[0] * size_pre[1]
            print('param num', num)
        return num

    def set_config(self, cfg: EasyDict, info_dict):
        self.rank = cfg.RANK
        self.omg_max = cfg.OMG_MAX
        self.omg_init = cfg.OMG_INIT
        self.event_score_mode = cfg.EVENT_SCORE
        self.embed_mode = cfg.EMBED_MODE
        if 'VECTOR_NORM' in cfg:
            self.vector_norm = cfg.VECTOR_NORM
        else:
            self.vector_norm = True

        self.sizes = cfg.sizes # [s, r, o, t]
        self.embed_dict = info_dict
 
    @staticmethod
    def has_time():
        return True

    def get_device(self):
        return list(self.modules())[-1].weight.device

    def embed_sub(self, x):
        c = self.ent_c(x)
        r, i = c[:, : self.rank], c[:, self.rank: ]
        s = torch.complex(r, i)
        return s

    def embed_obj(self, x):
        c = self.ent_c(x)
        r, i = c[:, : self.rank], c[:, self.rank: ]
        o = torch.complex(r, -i)
        return o

    def embed_pre(self, x):
        c = self.pre_c(x)
        r, i = c[:, : self.rank], c[:, self.rank: ]
        p = torch.complex(r, i)
        return p

    def embed_omg(self, batch_size):
        return self.tim_omg.repeat((batch_size, 1)) / self.time_cnt

    def embed_time(self, t):
        omg = self.embed_omg(t.shape[0])
        t = torch.complex(torch.cos(omg * t), torch.sin(omg * t))
        return t

    def embed_all(self, x):
        s = self.embed_sub(x[:, 0])
        r = self.embed_pre(x[:, 1])
        o = self.embed_obj(x[:, 2])
        return s, r, o

    def embed_rand(self, count, time_range=None):
        device = self.get_device()
        s = self.embed_sub(torch.randint(self.ent_cnt, (count[0], ), device=device))
        r = self.embed_pre(torch.randint(self.pre_cnt, (count[1], ), device=device))
        o = self.embed_obj(torch.randint(self.ent_cnt, (count[2], ), device=device))
        if time_range:
            t = self.embed_time(torch.randint(time_range[0], time_range[1], (count[3], 1), device=device))
        else:
            t = self.embed_time(torch.randint(self.time_cnt, (count[3], 1), device=device))
        return s, r, o, t

    def embed_event_type(self, x):
        # x: [s, p, o]
        if self.embed_mode == 'spo':
            idx_list = []
            spo_to_idx = self.embed_dict['spo']
            for i in range(x.shape[0]):
                s, p, o = int(x[i, 0]), int(x[i, 1]), int(x[i, 2])
                idx_list += [spo_to_idx[(s, p, o)]]
            idx_tensor = torch.LongTensor(idx_list).to(self.get_device())
            c = self.event_type_c(idx_tensor)
            r, i = c[:, : self.rank], c[:, self.rank :]
            return torch.complex(r, i)
        elif self.embed_mode == 'sp,po':
            sp_idx_list = []
            po_idx_list = []
            sp_to_idx = self.embed_dict['sp']
            po_to_idx = self.embed_dict['po']
            for i in range(x.shape[0]):
                s, p, o = int(x[i, 0]), int(x[i, 1]), int(x[i, 2])
                sp_idx_list += [sp_to_idx[(s, p)]]
                po_idx_list += [po_to_idx[(p, o)]]
            sp_tensor = torch.LongTensor(sp_idx_list).to(self.get_device())
            po_tensor = torch.LongTensor(po_idx_list).to(self.get_device())
            spc = self.sp_c(sp_tensor)
            poc = self.po_c(po_tensor)
            return torch.complex(spc[:, : self.rank], spc[:, self.rank :]) * \
                    torch.complex(poc[:, : self.rank], poc[:, self.rank :])
        elif self.embed_mode == 's,p,o':
            s = self.embed_sub(x[:, 0])
            p = self.embed_pre(x[:, 1])
            o = self.embed_obj(x[:, 2])
            return s * p * o

    def get_subjects(self):
        idx_list = torch.arange(0, self.ent_cnt, device=self.get_device())
        s = self.embed_sub(idx_list)
        return s

    def get_objects(self):
        idx_list = torch.arange(0, self.ent_cnt, device=self.get_device())
        o = self.embed_obj(idx_list)
        return o

    def get_times(self, time_range=None, reverse=False):
        if not reverse:
            if time_range:
                idx_list = torch.arange(time_range[0], time_range[1], device=self.get_device()).unsqueeze(1)
            else:
                idx_list = torch.arange(0, self.time_cnt, device=self.get_device()).unsqueeze(1)
        else:
            if time_range:
                idx_list = torch.arange(-time_range[1], -time_range[0], device=self.get_device()).unsqueeze(1)
            else:
                idx_list = torch.arange(-self.time_cnt + 1, 0, device=self.get_device()).unsqueeze(1)
        t = self.embed_time(idx_list)
        return t

    def get_query_times(self, time_range=None):
        f_idx_list = self.get_times(time_range)
        r_idx_list = self.get_times(time_range, reverse=True)
        # print('f_idx_list', f_idx_list.shape, 'r_idx_list', r_idx_list.shape)
        return torch.cat((r_idx_list, f_idx_list), 0)

    def get_reg(self, x):
        s, r, o, _ = self.embed_all(x)
        return math.pow(2, 1 / 3) * torch.sqrt(s ** 2 + 1e-9), \
                torch.sqrt(r ** 2 + 1e-9), \
                math.pow(2, 1 / 3) * torch.sqrt(o ** 2 + 1e-9)

    def mod(self, x, dim):
        s = x * torch.conj(x)
        return torch.sqrt(s.real.sum(dim) + 1e-9)

    def forward(self, x, time_range=None):
        x = x.to(self.get_device())
        lhs_t = self.embed_event_type(x[:, : 3])
        rhs_t = self.get_times(time_range)

        # print("Average lhs_t, rhs_t Mod", self.mod(lhs_t, 1).mean(), self.mod(rhs_t, 1).mean())

        score_t = 0
        if self.event_score_mode == 'l2':
            h = self.embed_h(x[:, : 3])
            dis = (lhs_t.unsqueeze(1) * rhs_t.unsqueeze(0) - h.unsqueeze(1)).abs().sum(2)
            score_t = -dis
        elif self.event_score_mode == 'cos':
            if self.vector_norm:
                lhs_t = lhs_t / (self.mod(lhs_t, 1).unsqueeze(1) + 1e-9)
                rhs_t = rhs_t / (self.mod(rhs_t, 1).unsqueeze(1) + 1e-9)
            score_t = (lhs_t @ rhs_t.t()).real

        return score_t

    def event_score(self, event):
        # s, r, o: int
        with torch.no_grad():
            x = self.to_batch_tensor(event)
            _ = torch.zeros(size=(x.shape[0], 1), dtype=torch.long, device=x.device)
            x = torch.cat((x, _), 1)
            score_t = self.forward(x)
        return score_t

    def ta_score(self, ta):
        # ta(normal): ndarray of [batch, 4], [[s, o, p1, p2]]
        # ta(generalized): ndarray of [batch, 6], [[s1, p1, o1, s2, p2, o2]]
        # print(f'calc rule curve, s:{s}, o:{o}, p1:{p1}, p2:{p2}')
        with torch.no_grad():
            x = self.to_batch_tensor(ta)
            t = self.get_query_times()
            e1 = self.embed_event_type(torch.stack((x[: , 0], x[: , 1], x[: , 2]), 1))
            e2 = self.embed_event_type(torch.stack((x[: , 3], x[: , 4], x[: , 5]), 1))

            lhs = e1.unsqueeze(1)
            rhs = e2.unsqueeze(1) * t.unsqueeze(0)

        if self.vector_norm:
            lhs = lhs / self.mod(lhs, 2).unsqueeze(2)
            rhs = rhs / self.mod(rhs, 2).unsqueeze(2)
            # print('vector normed')
        # score = (lhs * torch.conj(rhs)).real.sum(2)
        # score = -self.mod(lhs - rhs, 2) ** 2
        score = -((lhs - rhs).abs() ** 2).sum(2)
        # print(f'<{x}>: {score.max(1)}')

        # print("Average e1, e2, t Mod", self.mod(e1, 1).mean(), self.mod(e2, 1).mean(), self.mod(t, 1).mean())
        # print("Average lhs, rhs Mod", self.mod(lhs.squeeze(), 1).mean(), self.mod(rhs.squeeze(), 2).mean())

        return score.cpu().numpy()


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, scores, labels, time_range):
        num_classes = time_range[1] - time_range[0]
        scores = scores[: , time_range[0] : time_range[1]]
        labels = labels - time_range[0]
        # print('time_range: ', time_range)
        # print('labels: ', labels.min(), labels.max())
        assert ((labels >= 0) * (labels < num_classes)).all()
        one_hot = F.one_hot(labels, num_classes=num_classes)
        return F.cross_entropy(scores, one_hot.float())


class MSELoss(torch.nn.Module):
    def __init__(self, pos_score, neg_score):
        super().__init__()
        self.pos_score = pos_score
        self.neg_score = neg_score

    def forward(self, scores, labels, time_range):
        num_classes = time_range[1] - time_range[0]
        scores = scores[: , time_range[0] : time_range[1]]
        labels = labels - time_range[0]
        # print('labels: ', labels.min(), labels.max())
        assert ((labels >= 0) * (labels < num_classes)).all()
        one_hot = F.one_hot(labels, num_classes=num_classes)

        # return F.cross_entropy(scores, one_hot.float())

        # one_hot_b = self.blur_conv(one_hot.t().float().unsqueeze(1)).squeeze().t()
        one_hot_b = one_hot

        # l_pos = (scores[one_hot_b > 0].min() - pos_score) ** 2
        # l_neg = (scores[one_hot_b < 1].max() - neg_score) ** 2

        ground_truth = one_hot_b * self.pos_score + (1 - one_hot_b) * self.neg_score

        dist = ((scores - ground_truth) ** 2 + 1e-9)
        l_pos = (dist * one_hot).sum() / one_hot.sum()
        l_neg = (dist * (1 - one_hot)).sum() / (1 - one_hot).sum()

        # return dist.mean()

        return l_pos + l_neg