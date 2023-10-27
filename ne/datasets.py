# Copyright (c) Facebook, Inc. and its affiliates.
#

from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List

from sklearn.metrics import average_precision_score

import numpy as np
import torch
import random
import os
import math

from .utils import count_predicates


demo_name = 'PersonalDecision'


class TemporalDataset(object):
    def __init__(self, name: str, root_path: str):
        self.root = Path(root_path) / name
        self.queries = None
        self.general_queries = None
        self.part_queries = {}
        self.name = name
        print('Dataset:', name)

        if name == demo_name:
            in_file = str(self.root / ('data.txt'))
            self.data = self.read_data_txt(in_file)
            maxis = np.max(self.data, axis=0)
            self.n_entities = int(max(maxis[0], maxis[2]) + 1)
            self.n_predicates = int(maxis[1] + 1)
            self.n_timestamps = int(maxis[3] + 1)
            self.ent_to_id = self.read_ids(str(self.root / f'entity2id.txt'))
            self.rel_to_id = self.read_ids(str(self.root / f'relation2id.txt'))
            return

        for file in os.listdir(self.root):
            pre, suf = file.split('.')
            if pre == 'queries':
                self.queries = np.load(self.root / file)
            elif pre[: 8] == 'queries_':
                self.part_queries[ int(pre.replace('queries_', '')) ] = np.load(self.root / file)
            elif pre == 'general_queries':
                self.general_queries = np.load(self.root / file)
        
        print('Part Queries Keys:', list(self.part_queries.keys()))

        self.data = {}
        for f in ['train', 'test', 'valid']:
            if 'forecasting' in name:
                in_file = str(self.root / (f + '.txt'))
                self.data[f] = self.read_data_txt(in_file, 24)
            elif name == 'GDELT':
                in_file = str(self.root / (f + '.txt'))
                self.data[f] = self.read_data_txt(in_file, 15)
            else:
                in_file = open(str(self.root / (f + '.pickle')), 'rb')
                self.data[f] = pickle.load(in_file)

        maxis = np.max(np.vstack((self.data['train'], self.data['test'], self.data['valid'])), axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        # self.n_predicates *= 2
        if maxis.shape[0] > 4:
            self.n_timestamps = max(int(maxis[3] + 1), int(maxis[4] + 1))
        else:
            self.n_timestamps = int(maxis[3] + 1)
        try:
            inp_f = open(str(self.root / f'ts_diffs.pickle'), 'rb')
            self.time_diffs = torch.from_numpy(pickle.load(inp_f)).cuda().float()
            # print("Assume all timestamps are regularly spaced")
            # self.time_diffs = None
            inp_f.close()
        except OSError:
            print("Assume all timestamps are regularly spaced")
            self.time_diffs = None

        try:
            e = open(str(self.root / f'event_list_all.pickle'), 'rb')
            self.events = pickle.load(e)
            e.close()

            f = open(str(self.root / f'ts_id'), 'rb')
            dictionary = pickle.load(f)
            f.close()
            self.timestamps = sorted(dictionary.keys())
        except OSError:
            print("Not using time intervals and events eval")
            self.events = None

        if self.events is None:
            try:
                inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
                self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
                inp_f.close()
            except OSError:
                print("No to_skip file")
                self.to_skip = {'lhs': None, 'rhs': None}

        if 'forecasting' or 'GDELT' in name:
            self.ent_to_id = self.read_ids(str(self.root / f'entity2id.txt'))
            self.rel_to_id = self.read_ids(str(self.root / f'relation2id.txt'))
            self.time_to_id = list(range(self.n_timestamps))
        else:
            self.ent_to_id = self.read_ids(str(self.root / f'ent_id'))
            self.rel_to_id = self.read_ids(str(self.root / f'rel_id'))
            self.time_to_id = self.read_ids(str(self.root / f'ts_id'))

        print(f"train data shape: {self.data['train'].shape}")
        print(f"time stamps: {self.n_timestamps}")

        # If dataset has events, it's wikidata.
        # For any relation that has no beginning & no end:
        # add special beginning = end = no_timestamp, increase n_timestamps by one.

    def prep_for_special_baselines(self):
        print('Start prep for boxte')
        data_all = self.get_train().astype('long')
        positive_list = [(s, p, o, t) for (s, p, o, t) in data_all]
        self.positive_set = set(positive_list)
        print('Start prep for taster')
        self.get_mask_dict()

    def read_data_txt(self, file_name, interval=1):
        data_list = []
        with open(file_name) as f:
            for line in f.readlines():
                l = line.strip('\n\r').split('\t')
                data_list.append([int(l[0]), int(l[1]), int(l[2]), int(l[3]) // interval])
        return np.array(data_list)

    def read_ids(self, file_name):
        id_dict = {}
        with open(file_name, 'r') as f:
            for line in f.readlines():
                x, id = line.strip('\n\r').split('\t')
                id_dict[int(id)] = x
        return id_dict

    def has_intervals(self):
        return self.events is not None

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        if self.name == demo_name:
            return self.data
        if hasattr(self, 'data_all'):
            return self.data_all
        print(f"Train: {self.data['train'].shape}, Valid: {self.data['valid'].shape}, Test: {self.data['test'].shape}")
        tmin, tmax = self.data['train'][:, 3].min(), self.data['train'][:, 3].max()
        print('Train time', tmin, tmax)
        tmin, tmax = self.data['valid'][:, 3].min(), self.data['valid'][:, 3].max()
        print('Valid time', tmin, tmax)
        tmin, tmax = self.data['test'][:, 3].min(), self.data['test'][:, 3].max()
        print('Test time', tmin, tmax)
        data = np.vstack((self.data['train'], self.data['valid'], self.data['test']))
        self.data_all = data
        return data

    def sample_negative(self, data, rep_num=1, sample_type='s'):
        data = np.repeat(data, repeats=rep_num, axis=0)
        if sample_type == 's':
            data[:, 0] = np.random.randint(self.n_entities, size=data.shape[0])
        elif sample_type == 'p':
            data[:, 1] = np.random.randint(self.n_predicates, size=data.shape[0])
        elif sample_type == 'o':
            data[:, 2] = np.random.randint(self.n_entities, size=data.shape[0])
        elif sample_type == 't':
            data[:, 3] = np.random.randint(self.n_timestamps, size=data.shape[0])
        else:
            raise NotImplementedError(f"sample type '{sample_type}' not implemented")
        return data

    def sample_pure_negative(self, data, rep_num=1, sample_type='s'):
        data_shape = data.shape
        data = self.sample_negative(data, rep_num, sample_type)

        def find_positive(row):
            return np.array((row[0], row[1], row[2], row[3]) in self.positive_set)
        
        while True:
            pos_mask = np.apply_along_axis(find_positive, 0, data)
            if pos_mask.sum() == 0:
                break
            data_ = self.sample_negative(data, 1, sample_type)
            neg_mask = ~np.apply_along_axis(find_positive, 0, data_)
            add_len = min(pos_mask.sum(), neg_mask.sum())
            data[pos_mask][: add_len] = data_[neg_mask][: add_len]

        return np.reshape(data, (data_shape[0], rep_num, data_shape[1]))

    def iterate_taster(self, batch=512, sample_size=500, name='train', filter=True, eval=False):
        tensors = torch.tensor(self.get_train())
        # tensors = torch.tensor(self.data[name])
        tensors1 = tensors[torch.randperm(tensors.shape[0])] # shuffle
        tensors2 = tensors[torch.randperm(tensors.shape[0])] # shuffle

        n_batches = math.ceil(tensors.shape[0] / batch)
        if eval and name == 'train':
            n_batches = math.ceil(5000 / batch)

        fs = ['head', 'tail'] if eval else ['head', 'tail']
        for i in range(n_batches):
            for mask_part in fs:
                if not eval:
                    tensors = tensors1 if mask_part == 'tail' else tensors2
                elif name == 'train':
                    tensors = tensors[ : 5000]
                pos_batch = tensors[i * batch : min((i + 1) * batch, tensors.shape[0])]
                dates_id = pos_batch[ : , 3 : ]
                if eval:
                    facts = pos_batch[ : , : 4].unsqueeze(1)
                else:
                    facts = self.get_pos_neg_batch_taster(pos_batch, sample_size, mask_part=mask_part, filter=filter)
                
                # print(facts.shape, i, n_batches)
                    
                yield facts, mask_part, dates_id

    def get_pos_neg_batch_taster(self, pos_batch, sample_size, mask_part='head', filter=False):
        neg_samples = []
        rep_idx = 0 if mask_part == 'head' else 2

        neg_sample = np.random.randint(low=1, high=self.n_entities, size = (pos_batch.shape[0], sample_size * 2))
        neg_sample = (neg_sample + pos_batch[ : , rep_idx].unsqueeze(1).numpy()) % self.n_entities

        def sample(i):
            triplet = tuple(pos_batch[i].numpy())
            ht_mask = self.get_mask_key(triplet, mask_part=mask_part)
            assert triplet in self.positive_set
            mask = np.in1d(neg_sample[i], self.mask_dict[mask_part][ht_mask], assume_unique=True, invert=True)
            tmp_sample = neg_sample[i][mask]
            if tmp_sample.size < sample_size:
                # print('warning')
                tmp_sample = neg_sample[i]
            return tmp_sample[ : sample_size]

        if filter:
            for i in range(len(pos_batch)):
                neg_samples.append(sample(i))
            neg_sample = np.stack(neg_samples, axis=0)
        
        pos_facts = pos_batch[ : , : 3].unsqueeze(1)
        neg_facts = pos_facts.repeat(1, sample_size, 1)
        
        neg_facts[ : , : , rep_idx] = torch.LongTensor(neg_sample)[ : , : sample_size]

        return torch.cat((pos_facts, neg_facts), dim=1)

    def get_mask_key(self, triplet, mask_part='head'):
        triplet = tuple(triplet)
        return triplet[1 : ] if mask_part == 'head' else triplet[0 : 2] + triplet[3 : ]

    def get_mask_dict(self):
        print('start building mask dict')
        self.mask_dict = {'head' : {}, 'tail' : {}}
        for triplet in self.get_train():
            head_mask, tail_mask = self.get_mask_key(triplet), self.get_mask_key(triplet, mask_part='tail')
            if head_mask not in self.mask_dict['head']:
                self.mask_dict['head'][head_mask] = np.array([]).astype(int)
            tmp = self.mask_dict['head'][head_mask]
            self.mask_dict['head'][head_mask] = np.append(tmp, triplet[0])

            if tail_mask not in self.mask_dict['tail']:
                self.mask_dict['tail'][tail_mask] = np.array([]).astype(int)
            tmp = self.mask_dict['tail'][tail_mask]
            self.mask_dict['tail'][tail_mask] = np.append(tmp, triplet[2])
        
        print('finish building mask dict')

    def get_corrupted_train(self, data):
        data_cs = np.copy(data)
        data_co = np.copy(data)
        data_ct = np.copy(data)
        leng = data.shape[0]
        # sub_end, obj_end = leng // 3, leng * 2 //3
        data_cs[: , 0] = np.random.randint(self.n_entities, size=leng)
        data_co[: , 2] = np.random.randint(self.n_entities, size=leng)
        data_ct[: , 3] = np.random.randint(self.n_timestamps, size=leng)

        ones = np.ones((leng, 1))
        zeros = np.zeros((leng, 1))
        
        new_train = np.zeros((leng * 4, 5))
        rand_list = np.random.permutation(leng)
        idx_list = np.arange(leng)

        new_train[idx_list * 4] = np.concatenate((data[rand_list], ones), 1)
        new_train[idx_list * 4 + 1] = np.concatenate((data_cs[rand_list], zeros), 1)
        new_train[idx_list * 4 + 2] = np.concatenate((data_co[rand_list], zeros), 1)
        new_train[idx_list * 4 + 3] = np.concatenate((data_ct[rand_list], zeros), 1)

        return new_train

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps

    def get_ent_id(self):
        return self.ent_to_id
        
    def get_rel_id(self):
        return self.rel_to_id

    def get_time_id(self):
        return self.time_to_id

    def get_ent_id_list(self):
        return np.arange(self.n_entities).tolist()

    def get_rel_id_list(self):
        return np.arange(self.n_predicates).tolist()
    
    def get_timestamps(self):
        if self.name != demo_name:
            return np.vstack((self.data['train'], self.data['valid'], self.data['test']))[:, 3]
        else:
            return self.data[:, 3]
