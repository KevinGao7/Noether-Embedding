# Copyright (c) Facebook, Inc. and its affiliates.

from tqdm.auto import tqdm
import torch
from torch import nn
from torch import optim
import wandb
from copy import copy

from .models import TAModel, NE, BoxTE, TASTER
from .regularizers import Regularizer
from .datasets import TemporalDataset


class NEOptimizer(object):
    def __init__(
            self, model,
            optimizer: optim.Optimizer,
            time_loss,
            batch_size: int = 256,
            clip: float = 10,
            verbose: bool = True,
    ):
        self.model = model
        self.time_loss = time_loss
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.clip = clip
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor, time_range=None, first_epoch=False):
        # examples = examples[torch.randperm(examples.shape[0]), :]
        sum_t, sum_et = 0, 0
        total_len = examples.shape[0]

        with tqdm(list(range(0, total_len, self.batch_size))) as bar:
            for st in bar:
                ed = min(st + self.batch_size, total_len)
                input_batch = copy(examples[st : ed])
                score_t = self.model.forward(input_batch[:, : 3])
                l_t = self.time_loss(score_t, input_batch[:, 3], time_range)
                l = l_t
                
                self.optimizer.zero_grad()
                l.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                
                sum_t += l_t * (ed - st)
                bar.set_postfix(
                    l_t=f'{float(sum_t) / ed:.3f}'
                )
                
                wandb.log({
                    'loss_time': float(sum_t) / ed,
                    'loss_event_type': float(sum_et) / ed})

        return sum_t / examples.shape[0]


class TKBCOptimizer(object):
    def __init__(
            self, model: TAModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                predictions, factors, time = self.model.forward_(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.3f}',
                    reg=f'{l_reg.item():.3f}',
                    cont=f'{l_time.item():.3f}'
                )
        return l


class DEOptimizer(object):
    def __init__(
            self, model: TAModel,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.neg_ratio = model.neg_ratio
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
    
    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss_f = nn.CrossEntropyLoss()
        with tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                scores = self.model.forward_(input_batch)
                num_examples = int(scores.size(0) / (self.neg_ratio+1))
                scores_reshaped = scores.view(num_examples, self.neg_ratio+1)
                l = torch.zeros(num_examples).long().cuda()
                loss = loss_f(scores_reshaped, l)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{loss.item():.3f}'
                )
        return loss


class TEOptimizer(object):
    def __init__(
            self, model_name: str, model: TAModel,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model_name = model_name
        self.model = model
        self.neg_ratio = model.neg_ratio
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
    
    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        with tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                loss = self.model.log_rank_loss(input_batch) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.model_name == 'ATISE':
                    self.model.regularization_embeddings()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{loss.item():.3f}'
                )
        return loss


class BoxOptimizer(object):
    def __init__(
            self, model: BoxTE,
            optimizer: optim.Optimizer,
            cfg,
            verbose: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.dataset = cfg.dataset
        self.batch_size = cfg.BATCH_SIZE
        self.use_time_reg = cfg.use_time_reg
        self.num_negative_samples = cfg.num_negative_samples
        self.verbose = verbose
        self.loss_func = self.model.get_loss_func()
    
    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        total_loss = 0
        with tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                neg_sample = self.dataset.sample_pure_negative(input_batch.cpu().numpy(), self.num_negative_samples)
                neg_sample = torch.from_numpy(neg_sample).to(self.model.device)
                print('Remote POS, NEG', input_batch.shape, neg_sample.shape)
                # POS, NEG: [batch_size, nary], [batch_size, neg_num, nary]
                pos_sample, neg_sample = input_batch.unsqueeze(1).permute(1, 2, 0), neg_sample.permute(1, 2, 0)
                print('Transformed POS, NEG', pos_sample.shape, neg_sample.shape)
                # POS, NEG: [1, nary, batch_size], [neg_num, nary, batch_size]
                pos_emb, neg_emb = self.model.forward_(pos_sample, neg_sample)
                if self.use_time_reg:
                    loss = self.loss_func(pos_emb, neg_emb, time_bumps=self.model.compute_combined_timebumps(ignore_dropout=True))
                else:
                    loss = self.loss_func(pos_emb, neg_emb)
                self.optimizer.zero_grad()
                if not loss.isfinite():
                    print('Loss is {}. Skipping to next mini batch.'.format(loss.item()))
                    continue
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().item()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{loss.item():.3f}'
                )
        total_loss /= examples.shape[0]
        return total_loss


class TasterOptimizer(object):
    def __init__(
            self, model: TASTER,
            optimizer: optim.Optimizer,
            cfg,
            verbose: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.dataset = cfg.dataset
        self.batch_size = cfg.BATCH_SIZE
        self.sample_size = cfg.sample_size
        self.it = cfg.it
        self.sizes = cfg.sizes
        self.verbose = verbose
    
    def epoch(self, examples: torch.LongTensor):
        loss_f = nn.CrossEntropyLoss()
        total_loss = 0
        for facts, mask_part, date_ids in self.dataset.iterate_taster(batch=self.batch_size, sample_size=self.sample_size, filter=True):
            self.optimizer.zero_grad()

            facts = facts.cuda()
            [heads, rels, tails] = [facts[ : , : , i] for i in range(3)]

            if not self.it:
                date_ids = date_ids.cuda()
                scores = self.model.forward_(heads, rels, tails, date_ids, mask_part=mask_part)
            else:
                start_ids, end_ids = date_ids
                # print(start_ids.shape, heads.shape, rels.shape)
                start_ids, end_ids = start_ids.cuda(), end_ids.cuda()
                scores1 = self.model.forward_(heads, rels, tails, start_ids, mask_part=mask_part)
                scores2 = self.model.forward_(heads, rels + self.sizes[1] // 2, tails, end_ids, mask_part=mask_part)
                scores = 1 / 2 * (scores1 + scores2)
                
            l = torch.zeros(facts.shape[0]).long().cuda()
            loss = loss_f(scores, l) + self.model.reg_loss() + self.model.weight_loss()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.cpu().item()
        
        total_loss /= examples.shape[0]
        print("Loss in iteration : " + str(total_loss))
        return total_loss
