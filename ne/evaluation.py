from typing import Dict
import torch
import numpy as np
import seaborn as sns
import matplotlib
import math
import time
import random
import threading

from mpl_toolkits import mplot3d
from collections import defaultdict
matplotlib.use('agg')
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import copy
from sklearn import metrics

from .utils import *


thread_cnt = 1


def calc_gc(c_l):
    if c_l[2] == 0:
        return 0
    pre, rec = c_l[2] / c_l[0], c_l[2] / c_l[1]
    return 2 * pre * rec / (pre + rec + 1e-9)


def calc_gc_info(delta, dt, occur_s, occur_o):
    occur_s = sorted(occur_s)
    occur_o = sorted(occur_o)
    if len(occur_s) <= 1:
        return [len(occur_s), len(occur_o), 0]
    if len(occur_o) <= 1:
        return [len(occur_s), len(occur_o), 0]
        
    sp = 0
    j = 0
    for i in range(len(occur_s)):
        t1 = occur_s[i]
        while j < len(occur_o):
            t2 = occur_o[j]
            if t2 - t1 < dt - delta:
                j += 1
            else:
                if t2 - t1 <= dt + delta:
                    sp += 1
                    j += 1
                break

    return [len(occur_s), len(occur_o), sp]


def evaluate_ta(rule, tri_to_time, eta=0.1):
    s1, p1, o1, s2, p2, o2, dt = rule
    delta = math.ceil(abs(dt) * eta)
    gc_info = calc_gc_info(delta, dt, tri_to_time[(s1, p1, o1)], tri_to_time[(s2, p2, o2)])
    return gc_info


def get_query_size(query, tri_to_time):
    s1, p1, o1, s2, p2, o2 = query[: 6]
    time1 = tri_to_time[(s1, p1, o1)]
    time2 = tri_to_time[(s2, p2, o2)]
    return len(time1) + len(time2)


class ThreadForSegProcess(threading.Thread):
    def __init__(self, thread_cnt, total_size, thread_idx, func, **args) -> None:
        super().__init__()
        self.thread_cnt = thread_cnt
        self.total_size = total_size
        self.thread_idx = thread_idx
        self.thread_size = math.ceil(self.total_size / self.thread_cnt)
        self.func = func
        self.args = args
        self.results = []

    def run(self):
        start_time = time.time()
        print(f'Thread@{self.thread_idx} start, len: {len(self.results)}')
        st = self.thread_idx * self.thread_size
        ed = min(st + self.thread_size, self.total_size)
        for i in range(st, ed):
            self.results.extend(self.func(i, **self.args))
            if (i - st) % 100 == 0:
                print(f'Thread@{self.thread_idx}, {(i - st) / (ed - st) * 100}% ({i} in [{st}, {ed}]), Time used: {time.time() - start_time}')
        print(f'Thread@{self.thread_idx} finish, len: {len(self.results)}')


def multi_thread_work(thread_cnt, total_size, func, **args):
        thread_list = []
        results = []
        for i in range(thread_cnt):
            thread = ThreadForSegProcess(thread_cnt, total_size, i, func, **args)
            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
            results.extend(thread.results)
        return results


class TemporalAssociationLearning():
    def __init__(self, cfg, dataset) -> None:
        self.INVALID_TAU = -100007
        self.dataset = dataset
        # self.query_num = cfg.QUERY_NUM
        # self.query_size = cfg.QUERY_SIZE
        # self.query_max_tau = cfg.QUERY_MAX_TAU
        self.g_th = cfg.SCORE_THRESHOLD
        # self.pos_ratio = cfg.QUERY_POS_RATIO
        self.eta = cfg.ETA
        self.freq_thd = cfg.FREQ_THRESHOLD
        self.task_name = cfg.TASK_NAME
        self.ta_score_mode = cfg.TA_SCORE

        self.neg_gc_thd = cfg.NEGATIVE_GC_THRESHOLD
        self.pos_gc_thd = cfg.POSTIVE_GC_THRESHOLD
        self.data_range = (cfg.EVAL_DATE_RANGE.ST, cfg.EVAL_DATE_RANGE.ED)
        self.cfg = cfg
        
        if 'USE_GENERAL_RULE' in cfg and cfg['USE_GENERAL_RULE'] == True:
            self.rule_type = 'general'
        else:
            self.rule_type = 'specific'
        print('rule_type', self.rule_type)

        if self.task_name == 'mining':
            self.event_num_differ = cfg.EVENT_NUM_DIFFERENT_FACTOR
        else:
            self.event_num_differ = 10

        train_data = dataset.get_train().astype('int64')
        self.sizes = dataset.get_shape()

        # self.train_data = train_data[self.valid_data(train_data)]
        self.train_data = train_data

        self.train_tri_dict = count_triples(self.train_data)
        self.tri_to_time = self.train_tri_dict['tri_to_time']
        self.tri_to_freq = self.train_tri_dict['tri_to_freq']
        
        total_queries = self.get_total_queries(self.train_data, self.train_tri_dict)
        self.queries = self.split_queries(total_queries, self.train_tri_dict)

        print('Data Range: ', self.data_range)

    def valid_data(self, train_data):
        valid_list = np.logical_and((train_data[:, 3] >= self.data_range[0]), (train_data[:, 3] < self.data_range[1]))
        return valid_list

    def valid_queries(self, queries, tri_dict):
        tri_to_freq = tri_dict['tri_to_freq']
        valid_list = np.zeros(queries.shape[0]).astype(bool)
        for i in range(queries.shape[0]):
            q = queries[i].astype(int)
            t1, t2 = tri_to_freq[(q[0], q[1], q[2])], tri_to_freq[(q[3], q[4], q[5])]

            if t1 >= self.freq_thd and t2 >= self.freq_thd:
                # and abs(t1 - t2) / max(t1, t2) <= self.event_num_differ:
                valid_list[i] = True
        return valid_list

    def get_pos_queries(self, queries):
        pos_queries = queries[ queries[:, 7] >= self.pos_gc_thd ]
        return pos_queries

    def get_neg_queries(self, queries):
        neg_queries = queries[ queries[:, 7] <  self.neg_gc_thd ]
        neg_queries[:, 6] = self.INVALID_TAU
        return neg_queries

    def split_queries(self, queries, tri_dict):
        print('# query: ', queries.shape)

        if 'IS_DEMO' not in self.cfg or not self.cfg.IS_DEMO:
            valid_list = self.valid_queries(queries, tri_dict)
            queries = queries[valid_list]
        else:
            return queries

        print('# valid query: ', queries.shape)
        np.random.shuffle(queries)
        pos_queries = self.get_pos_queries(queries)
        neg_queries = self.get_neg_queries(queries)
        print(f'#Pos Queries: tau > 0: ({np.sum(np.abs(pos_queries[:, 6]) > 1)}), tau == 0: ({np.sum(np.abs(pos_queries[:, 6]) <= 1)})')

        if self.task_name == 'detection' or self.task_name == 'mining':
            print(f'#Queries: pos({len(pos_queries)}), neg({len(neg_queries)})')
            queries = np.concatenate((pos_queries, neg_queries), 0)
            return queries
        elif self.task_name == 'query':
            pos_queries = pos_queries[ pos_queries[:, 6] != 0 ]
            queries = pos_queries
            print(f'#Queries: {len(queries)}')
            return queries

    def get_total_queries(self, data, tri_dict):
        if 'IS_DEMO' in self.cfg and self.cfg.IS_DEMO:
            queries = []
            tri_to_freq = tri_dict['tri_to_freq']
            so_to_p = count_so_p(data)
            for so1, p1_set in tqdm(list(so_to_p.items())):
                # if len(queries) > 100000:
                #     break
                print('sop1', so1, p1_set)
                s1, o1 = so1
                for p1 in p1_set:
                    for so2, p2_set in so_to_p.items():
                        s2, o2 = so2
                        p2_set = so_to_p[so2]
                        print('sop2', so2, p2_set)
                        for p2 in p2_set:
                            print(f'Data ({s1, p1, o1, s2, p2, o2})')
                            queries += [[s1, p1, o1, s2, p2, o2, self.INVALID_TAU, 0]]
            queries = np.array(queries)
            return queries
        elif self.rule_type == 'general':
            # if self.dataset.general_queries is not None:
            if False:
                queries = self.dataset.general_queries
            else:
                tri_to_freq = tri_dict['tri_to_freq']
                so_to_p = count_so_p(data)
                s_to_o = count_s_o(data)
                print(f"so num: {len(list(so_to_p.keys()))}")
                sop_cnt = 0
                for y in so_to_p.values():
                    sop_cnt += len(y)
                print(f"sop num: {sop_cnt}")
                queries = []
                rand_so_to_p_list = list(so_to_p.items())
                random.shuffle(rand_so_to_p_list)
                for so1, p1_set in tqdm(rand_so_to_p_list):
                    # if len(queries) > 100000:
                    #     break
                    s1, o1 = so1
                    for p1 in p1_set:
                        if tri_to_freq[(s1, p1, o1)] >= self.freq_thd:
                            for so2, p2_set in rand_so_to_p_list:
                                s2, o2 = so2
                                p2_set = so_to_p[(s2, o2)]
                                if s1 == s2 or s1 == o2 or o1 == s2 or o1 == o2:
                                    for p2 in p2_set:
                                        if s1 == s2 and p1 == p2 and o1 == o2:
                                            continue
                                        if tri_to_freq[(s2, p2, o2)] >= self.freq_thd:
                                            queries += [[s1, p1, o1, s2, p2, o2]]
                queries = np.array(queries)
                np.save(self.dataset.root / 'general_queries.npy', queries)
        else:
            if self.data_range[1] >= self.sizes[3] and self.dataset.queries is not None:
                queries = self.dataset.queries
            elif int(self.data_range[1]) in self.dataset.part_queries:
                queries = self.dataset.part_queries[int(self.data_range[1])]
            else:
                tri_to_freq = tri_dict['tri_to_freq']
                so_to_p = count_so_p(data)
                so_to_p_list = list(so_to_p.items())
                so_to_p_len = len(so_to_p_list)

                sop_cnt = 0
                for y in so_to_p.values():
                    sop_cnt += len(y)

                print(f"so num: {len(list(so_to_p.keys()))}")
                print(f"sop num: {sop_cnt}")

                print('so_to_p_len', so_to_p_len)
                # so_to_p_len = 100

                def get_valid_rule(i):
                    tmp_queries = []
                    so, p_set = so_to_p_list[i]
                    s, o = so
                    p2_set = so_to_p[(o, s)]
                    for p1 in p_set:
                        if tri_to_freq[(s, p1, o)] >= self.freq_thd:
                            for p2 in p_set:
                                if p1 != p2 and tri_to_freq[(s, p2, o)] >= self.freq_thd:
                                    tmp_queries += [[s, p1, o, s, p2, o, self.INVALID_TAU, 0]]
                            for p2 in p2_set:
                                if tri_to_freq[(o, p2, s)] >= self.freq_thd:
                                    tmp_queries += [[s, p1, o, o, p2, s, self.INVALID_TAU, 0]]
                    return tmp_queries

                queries = multi_thread_work(thread_cnt, so_to_p_len, get_valid_rule)
                queries = np.array(queries)

                print(f'#Queries: {queries.shape}')
                
                np.random.shuffle(queries)
                # queries = queries[: 100000]
                valid_list = self.valid_queries(queries, tri_dict)
                queries = queries[valid_list]
                
                search_res, run_time = self.run_search(queries)
                results = search_res

                queries = np.concatenate((queries[:, : 6], results), 1)

                if self.data_range[1] >= self.sizes[3]:
                    np.save(self.dataset.root / 'queries.npy', queries)
                else:
                    np.save(self.dataset.root / f'queries_{int(self.data_range[1])}.npy', queries)

        return queries

    def get_ta_ce_ratio(self, queries, tri_to_time):
        ce_ratios = []
        for q in queries:
            s1, p1, o1, s2, p2, o2 = q[: 6]
            c = float(len(tri_to_time[(s1, p1, o1)]))
            e = float(len(tri_to_time[(s2, p2, o2)]))
            r = c / e if c < e else e / c
            ce_ratios += [r]
        return np.array(ce_ratios)

    def calc_gc_list(self, rule_list):
        gc_info_list = list()
        gc_list = np.zeros(rule_list.shape[0])
        for i in range(rule_list.shape[0]):
            if rule_list[i, 6] != self.INVALID_TAU:
                gc_info = evaluate_ta(rule_list[i, : 7], self.tri_to_time, eta=self.eta)
                gc_ = calc_gc(gc_info)
                gc_list[i] = gc_
                gc_info_list += [gc_info]
        return gc_list, gc_info_list

    def evaluate_model(self, model, queries):
        bs         = 100
        res_list   = np.zeros((queries.shape[0]))
        score_list = np.zeros((queries.shape[0]))
        thd        = self.g_th
        print(f'queries shape: {queries.shape}')
        print(f'thd: ', thd)
        for bi in range(math.ceil(queries.shape[0] / bs)):
            st, ed = bi * bs, min((bi + 1) * bs, queries.shape[0])
            ex = queries[st : ed]
            time_end = self.data_range[1]
            # print('ex', ex, rule_type)
            # s, r = model.ta_query(ex, ta_score_mode=self.ta_score_mode, time_range=(-time_end, time_end))
            s, r = model.ta_query(ex, ta_score_mode=self.ta_score_mode, time_range=None)
            score_list[st : ed], res_list[st : ed] = s.astype(float), r.astype(int)

        rule_list = np.concatenate((queries[:, : 6], res_list.astype(int)[:, np.newaxis] - self.sizes[3] + 1), 1)

        return score_list, rule_list

    def rule_structure(self, rule):
        s1, p1, o1, s2, p2, o2, t = rule[: 7]
        if t < 0:
            s1, p1, o1, s2, p2, o2 = s2, p2, o2, s1, p1, o1

        if s1 == s2 and o1 == o2:
            return 'AB->AB'
        if s1 == o2 and o1 == s2:
            return 'AB->BA'
        if s1 == s2:
            return 'AB->AC'
        if s1 == o2:
            return 'AB->CA'
        if o1 == s2:
            return 'AB->BC'
        if o1 == o2:
            return 'AB->CB'
        return '*'

    def evaluate(self, model):
        ## Run Model
        model = model.cuda()
        model.eval()
        start_time = time.time()
        score_list, rule_list = self.evaluate_model(model, self.queries)
        run_time = time.time() - start_time
        print('run time:', run_time)

        if 'IS_DEMO' in self.cfg and self.cfg.IS_DEMO:
            return score_list

        if self.task_name == 'mining':
            gc_list, gc_info_list = self.calc_gc_list(rule_list)
            gc_all = gc_list.mean()
            ta_cnt = (gc_list > self.pos_gc_thd).sum()

            for i in range(rule_list.shape[0]):
                if gc_list[i] >= self.pos_gc_thd:
                    self.show_ta_sample(self.tri_to_time, rule_list[i], \
                                        self.dataset.get_ent_id(), self.dataset.get_rel_id())
            
            
            metric_dict = {
                            'gc_all': gc_all,
                            'ta_cnt': ta_cnt
                        }
        else:
            ## Calc metircs
            gc_list, gc_info_list = self.calc_gc_list(rule_list)
            gc_all = gc_list.mean()
            gc_pos = gc_list.sum() / ((score_list > self.g_th).sum() + 1e-9)
            ta_cnt = (gc_list > self.pos_gc_thd).sum()

            if self.task_name == 'query':
                r_list = gc_list / self.queries[: , 7]
                r = r_list.mean()
                metric_dict = {
                                'gc_all': gc_all,
                                'r': r
                            }
            else:
                labels = (self.queries[:, 6] != self.INVALID_TAU)
                preds = score_list > self.g_th
                auc = metrics.roc_auc_score(labels, score_list)
                pre, rec, threshold = metrics.precision_recall_curve(labels, score_list)
                f1_total = (2 * pre * rec) / (pre + rec + 1e-9)
                
                opt_idx = np.argmax(f1_total)
                opt_pre, opt_rec, opt_thd, opt_f1 = pre[opt_idx], rec[opt_idx], threshold[opt_idx], f1_total[opt_idx]
                f1 = metrics.f1_score(labels, preds)
                metric_dict = {
                                'opt_f1': opt_f1,
                                'opt_pre': opt_pre,
                                'opt_rec': opt_rec,
                                'opt_thd': opt_thd,
                                'f1': f1,
                                'auc': auc,
                                'gc_all': gc_all,
                                'ta_cnt': ta_cnt
                            }
                model_pos_num = (score_list >= opt_thd).sum()
                model_neg_num = (score_list < opt_thd).sum()
                print('model_pos_num, model_neg_num', model_pos_num, model_neg_num)
                self.opt_thd = opt_thd

        print(metric_dict)
        wandb.log(metric_dict)

    def run_search(self, queries):
        time_start = time.time()
        results = []
        query_len = len(queries)
        def work(i):
            query = queries[i]
            gc, res = 0, self.INVALID_TAU
            r = query[: 7].copy()
            for dt in range(0, self.sizes[3]):
                for p in ['pos', 'neg']:
                    r[6] = dt if p == 'pos' else -dt
                    gc_info = evaluate_ta(r, self.tri_to_time, eta=self.eta)
                    gc_ = calc_gc(gc_info)
                    if gc_ > gc:
                        gc, res = gc_, dt
                    if gc_ >= 0.99:
                        break
            return [[res, gc]]
        results = multi_thread_work(thread_cnt, query_len, work)
        results = np.array(results)
        run_time = time.time() - time_start
        return results, run_time

    def analyze(self, model):
        print('begin analyze')
        query_size_list = []
        tau_list = []
        model = model.cuda()
        score_list, rule_list = self.evaluate_model(model, self.queries)
        gc_list, gc_info_list = self.calc_gc_list(rule_list)
        sp_list = [info[2] for info in gc_info_list]

        for q in self.queries:
            query_size_list += [get_query_size(q, self.tri_to_time)]
            tau_list += [q[6]]

        qsl = np.array(query_size_list)
        q_width = 2
        q_len = 40
        q_bins = np.linspace(0, q_len, int(q_len / q_width) + 1)
        qi_list = np.digitize(qsl, q_bins)
        qi_count = np.bincount(qi_list)
        print('query sizes: ', np.unique(qsl))
        print('query bins: ', q_bins)
        print('qi_count', qi_count)
        
        tl = np.array(tau_list)
        t_width = 10
        t_bins = np.linspace(-365, 365, int((365 * 2) / t_width) + 1)
        ti_list = np.digitize(tl, t_bins)
        ti_count = np.bincount(ti_list)
        print('tau sizes: ', np.unique(tl))
        print('tau bins: ', np.unique(t_bins))

        fig_dir = wandb.run.dir
        font_size_1 = 20
        font_size_2 = 18

        def mean_over_bins(idx_list, v_list, bins):
            res = [v_list[idx_list == i].mean() for i in range(0, len(bins))]
            return np.nan_to_num(res)
        
        def f1_over_bins(idx_list, labels, preds, bins):
            res = [metrics.f1_score(labels[idx_list == i], preds[idx_list == i]) for i in range(0, len(bins))]
            return np.nan_to_num(res)

        if self.task_name == 'query':
            r_list = gc_list / self.queries[:, 7]

            r_q_list = mean_over_bins(qi_list, r_list, q_bins)
            r_t_list = mean_over_bins(ti_list, r_list, t_bins)

            plot_bar(q_bins, r_q_list, f'{fig_dir}/r_qn.jpg', q_width - 0.5, \
                        'Number of Events', 'r', q_len, 1, font_size_1, font_size_2)

            plot_bar(t_bins, r_t_list, f'{fig_dir}/r_t.jpg', t_width - 0.5, \
                        'Relative Time (Day)', 'r', tl.max(), 1, font_size_1, font_size_2, tl.min() - 1)

            gc_bins = np.linspace(0, 1, 11)
            gc_digt = np.digitize(gc_list, gc_bins)
            gtgc_digt = np.digitize(self.queries[:, 7], gc_bins)

            print('gc count', np.bincount(gc_digt))
            print('gtgc count', np.bincount(gtgc_digt))
        else:
            labels = self.queries[:, 6] != self.INVALID_TAU
            preds = score_list >= self.g_th

            print('Total F1:', metrics.f1_score(labels, preds))
            
            f1_q_list = f1_over_bins(qi_list, labels, preds, q_bins)
            f1_t_list = f1_over_bins(ti_list, labels, preds, t_bins)
            
            print('F1 list:', f1_q_list)

            for i in range(len(q_bins)):
                pos_size = labels[qi_list == i].sum()
                tot_size = (qi_list == i).sum()

                print('q', q_bins[i], pos_size, tot_size)

            plot_scatter(self.queries[:, 7], -score_list, f'{fig_dir}/gc-g.jpg', \
                            'Maximal gc', 'Minimal g', font_size_1, font_size_2)

            for i in range(1, len(q_bins)):
                idx = qi_list == i
                plot_scatter(self.queries[:, 7][idx], score_list[idx], f'{fig_dir}/event_num-{q_bins[i - 1]}-gc-g.jpg', \
                                'Maximal gc', 'Minimal Energy', font_size_1, font_size_2)

            plot_bar(q_bins, f1_q_list, f'{fig_dir}/f1_qn.jpg', q_width - 0.5, \
                        'Number of Events', 'F1', q_len, 1, font_size_1, font_size_2)

    def show_ta_sample(self, tri_to_time, ta, ent_to_id, rel_to_id):
        if ta.shape[0] >= 7:
            s1, p1, o1, s2, p2, o2, dt = ta[: 7]
        else:
            s1, p1, o1, s2, p2, o2 = ta[: 6]
            dt = -1
        gc_info = evaluate_ta(ta[: 7], tri_to_time)

        times1 = sorted(tri_to_time[(s1, p1, o1)])
        times2 = sorted(tri_to_time[(s2, p2, o2)])

        gc = calc_gc(gc_info)
        print("====================================")
        print('Rule Sturcture', self.rule_structure(ta))
        print('gc info', gc_info, 'gc', gc)
        print(f"TA number: ({s1}, {p1}, {o1}), ({s2}, {p2}, {o2}), {dt}")
        print(f"TA meaning: ({ent_to_id[s1]}, {rel_to_id[p1]}, {ent_to_id[o1]}), ({ent_to_id[s2]}, {rel_to_id[p2]}, {ent_to_id[o2]}), {dt}")

        if len(times1) > 0 or len(times2) > 0:
            print(f'({ent_to_id[s1]}, {ent_to_id[o1]}): {times1}, ({ent_to_id[s2]}, {ent_to_id[o2]}): {times2}')
        print("====================================")
