from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib.pyplot as plt
import argparse
import math
import os
import random
import wandb

from sklearn.manifold import TSNE

from ne.models import NE


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def generate_events_for_ta(sizes, ta_desc):
    s, o, p1, p2, dt, num = ta_desc
    data = np.zeros((num * 2, 4))
    for i in range(num):
        if dt > -1:
            t0 = random.randint(0, sizes[3] - dt - 1)
            t1 = t0 + dt
        else:
            t0 = random.randint(0, sizes[3] - 1)
            t1 = random.randint(0, sizes[3] - 1)
        data[i, 0] = s
        data[i, 1] = p1
        data[i, 2] = o
        data[i, 3] = t0
        
        data[i + num, 0] = s
        data[i + num, 1] = p2
        data[i + num, 2] = o
        data[i + num, 3] = t1

    return data


def generate_events_for_tas(sizes, ta_desc_list):
    data = None
    ## ta_desc: (s, o, p1, p2, dt, num)
    for desc in ta_desc_list:
        artificial = generate_events_for_ta(sizes, desc)
        if data is None:
            data = artificial
        else:
            data = np.vstack((data, artificial))
    return data


def get_valid_so(so_to_freq, thd):
    s_list, o_list = [], []
    for (s, o), freq in so_to_freq.items():
        if freq >= thd:
            s_list += [s]
            o_list += [o]
    return np.stack((np.array(s_list), np.array(o_list)))


def get_valid_pre_pairs(tri_to_freq, thd):
    so_to_p = defaultdict(list)
    pairs = set()
    for (s, p, o), freq in tri_to_freq.items():
        if freq < thd:
            continue
        so_to_p[(s, o)] += [p]
    for (s, o), p_list in so_to_p.items():
        for i in range(len(p_list)):
            for j in range(i + 1, len(p_list)):
                pairs.add((p_list[i], p_list[j]))
                pairs.add((p_list[j], p_list[i]))
    
    p1_list, p2_list = [], []
    for (p1, p2) in pairs:
        p1_list += [p1]
        p2_list += [p2]
    return np.stack((np.array(p1_list), np.array(p2_list)))


def count_s_o(events):
    s_to_o = defaultdict(set)
    for e in events:
        s, o = int(e[0]), int(e[2])
        s_to_o[s].add(o)
    return s_to_o


def count_so(events):
    so_to_freq = defaultdict(int)
    for e in events:
        s, o = int(e[0]), int(e[2])
        so_to_freq[(s, o)] += 1
    return so_to_freq


def count_so_p(events):
    so_to_p = defaultdict(set)
    for e in events:
        s, p, o = int(e[0]), int(e[1]), int(e[2])
        so_to_p[(s, o)].add(p)
    return so_to_p


def count_predicates(events):
    pred_to_freq = defaultdict(int)
    for e in events:
        pred_to_freq[ e[1] ] += 1

    return pred_to_freq


def count_triples(events):
    events = sorted(events.tolist())
    tri_to_idx = defaultdict(int)
    tri_to_freq = defaultdict(int)
    tri_to_time = defaultdict(list)
    freq_to_tri = defaultdict(list)
    idx_cnt = 0

    for e in events:
        x = int(e[0]), int(e[1]), int(e[2])
        if x not in tri_to_idx.keys():
            tri_to_idx[x] = idx_cnt
            idx_cnt += 1
        tri_to_freq[x] += 1
        tri_to_time[x] += [int(e[3])]

    for e, f in tri_to_freq.items():
        freq_to_tri[f] += [e]

    return {'tri_to_freq': tri_to_freq, 'freq_to_tri': freq_to_tri, 'tri_to_time': tri_to_time, 'tri_to_idx': tri_to_idx}


def count_entities(events):
    ent_to_freq = defaultdict(int)
    freq_to_ent = defaultdict(list)

    for e in events:
        ent_to_freq[ int(e[0]) ] += 1
        ent_to_freq[ int(e[2]) ] += 1

    for e, f in ent_to_freq.items():
        freq_to_ent[f] += [e]

    return {'ent_to_freq': ent_to_freq, 'freq_to_ent': freq_to_ent}


def count_sp(events):
    sp_cnt = 0
    sp_to_idx = defaultdict(int)
    for e in events:
        sp = (int(e[0]), int(e[1]))
        if sp not in sp_to_idx:
            sp_to_idx[sp] = sp_cnt
            sp_cnt += 1
    return sp_to_idx


def count_po(events):
    po_cnt = 0
    po_to_idx = defaultdict(int)
    for e in events:
        po = (int(e[1]), int(e[2]))
        if po not in po_to_idx:
            po_to_idx[po] = po_cnt
            po_cnt += 1
    return po_to_idx


def get_shape(events):
    s1 = max(np.max(events[:, 0]), np.max(events[:, 2])) + 1
    s2 = np.max(events[:, 1]) + 1
    s3 = np.max(events[:, 3]) + 1
    return (s1, s2, s1, s3)


def plot_curve(bins, values, file_name, xl, yl, fs1, fs2):
    plt.figure()
    plt.plot(bins, values)
    plt.xlabel(xl, fontsize=fs1, fontweight='bold')
    plt.ylabel(yl, fontsize=fs1, fontweight='bold')
    plt.tick_params(labelsize=fs2)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def plot_bar(bins, values, file_name, width, xl, yl, xmax, ymax, fs1, fs2, xmin=0):
    plt.figure()
    plt.bar(bins - width / 2 - 0.5, values, width = width)
    plt.xlabel(xl, fontsize=fs1, fontweight='bold')
    plt.ylabel(yl, fontsize=fs1, fontweight='bold')
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.tick_params(labelsize=fs2)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()


def plot_scatter(x, y, file_name, xl, yl, fs1, fs2):
    plt.figure()
    plt.scatter(x, y, s=1)
    plt.xlabel(xl, fontsize=fs1, fontweight='bold')
    plt.ylabel(yl, fontsize=fs1, fontweight='bold')
    plt.tick_params(labelsize=fs2)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()


def plot_time_curve(time_scores, file_name, color='b', y_max=-1):
    plt.figure()
    plt.plot(np.arange(0, time_scores.shape[0]), time_scores, c = color)
    # if y_max > 0:
    #     plt.ylim((0, y_max))
    # else:
    #     y_max = time_scores.max()
    #     plt.ylim((0, y_max))
    plt.savefig(f'{file_name}')
    plt.close()


def plot_occur(occur_1, occur_2, file_name):
    plt.figure()
    plt.scatter(occur_1, [2] * len(occur_1), c='r')
    plt.scatter(occur_2, [1] * len(occur_2), c='b')
    plt.xlim((0, 365))
    plt.savefig(f'{file_name}')
    plt.close()


def plot_rules(model, rules, tri_to_time, ta_score_mode):
    ## rule: (s, o, r1, r2, dt, ...)
    print(rules)
    fig_dir = wandb.run.dir + '/figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    data_dir = wandb.run.dir + '/data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for rule in rules:
        s1, p1, o1, s2, p2, o2, dt = rule[: 7]
        s1, p1, o1, s2, p2, o2, dt = int(s1), int(p1), int(o1), int(s2), int(p2), int(o2), int(dt)
        if ta_score_mode == 'corr':
            rule_scores = model.ta_score_corr(np.array([[s1, p1, o1, s2, p2, o2]])).squeeze()
        else:
            rule_scores = model.ta_score(np.array([[s1, p1, o1, s2, p2, o2]])).squeeze()
        print('Rule: ', s1, p1, o1, s2, p2, o2, dt, 'max point: ', rule_scores.argmax())

        if isinstance(model, NE):
            with torch.no_grad():
                e1, e2 = model.to_batch_tensor([[s1, p1, o1]]), model.to_batch_tensor([[s2, p2, o2]])
                emb1, emb2 = model.embed_event_type(e1), model.embed_event_type(e2)
                emb1, emb2 = torch.cat((emb1.real, emb1.imag), 1), torch.cat((emb2.real, emb2.imag), 1)
                emb1, emb2 = emb1.cpu().numpy(), emb2.cpu().numpy()
            np.save(f'{data_dir}/event_emb_({s1}, {p1}, {o1}).npy', emb1)
            np.save(f'{data_dir}/event_emb_({s2}, {p2}, {o2}).npy', emb2)

        plot_time_curve(-rule_scores, f'{fig_dir}/rule_score_({s1}, {p1}, {o1}, {s2}, {p2}, {o2}, {dt}).png')
        np.save(f'{data_dir}/rule_score_({s1}, {p1}, {o1}, {s2}, {p2}, {o2}, {dt}).npy', rule_scores)
        # plot_time_curve(np.softmax(rule_scores, 0), f'{fig_dir}/rule_score_exp_({s1}, {p1}, {o1}, {r2}, {dt}).png', color=color, y_max=1)
        x = model.event_score(np.array([[s1, p1, o1]])).squeeze().cpu().numpy()
        # print('here', x)

        time_scores_1 = model.event_score(np.array([[s1, p1, o1]])).squeeze().cpu().numpy()
        plot_time_curve(time_scores_1, f'{fig_dir}/event_score_({s1}, {p1}, {o1}).png', y_max=-1)
        np.save(f'{data_dir}/event_score_({s1}, {p1}, {o1}).npy', time_scores_1)

        time_scores_2 = model.event_score(np.array([[s2, p2, o2]])).squeeze().cpu().numpy()
        plot_time_curve(time_scores_2, f'{fig_dir}/event_score_({s2}, {p2}, {o2}).png')
        np.save(f'{data_dir}/event_score_({s2}, {p2}, {o2}).npy', time_scores_2)

        occur_1 = tri_to_time[(s1, p1, o1)]
        occur_2 = tri_to_time[(s2, p2, o2)]
        plot_occur(occur_1, occur_2, f'{fig_dir}/event_occur_({s1}, {p1}, {o1}, {s2}, {p2}, {o2}).png')
        np.save(f'{data_dir}/event_occur_({s1}, {p1}, {o1}).npy', occur_1)
        np.save(f'{data_dir}/event_occur_({s2}, {p2}, {o2}).npy', occur_2)
        
        print(f"Occurs ({s1}, {p1}, {o1}, {s2}, {p2}, {o2}):")
        print(occur_1)
        print(occur_2)
        print("")

    # wandb.save(fig_dir)


def plot_demo_heatmap(heatmap, ticks, fs1, fs2, file_name='demo_score', cmap='Greys'):
    fig = plt.figure(figsize=(7.2, 6))
    ax = fig.add_subplot(111)
    max_v, min_v = heatmap.max(), heatmap.min()
    heatmap = (heatmap - min_v) / (max_v - min_v)
    for i in range(heatmap.shape[0]):
        heatmap[i, i] = 0
    im = ax.imshow(heatmap, cmap=cmap)
    cb = plt.colorbar(im, fraction=0.05, pad=0.05)
    # ax.set_xticks([0, 1, 2, 3, 4], ticks)
    # ax.set_yticks([0, 1, 2, 3, 4], ticks)
    # ax.tick_params(labelsize=fs2)
    ax.plot([-0.5, 1, 2, 3, 4.5], [-0.5, 1, 2, 3, 4.5], color='black')
    plt.xticks([0, 1, 2, 3, 4], ticks, fontsize=fs2, fontweight='bold')
    plt.yticks([0, 1, 2, 3, 4], ticks, fontsize=fs2, fontweight='bold')
    plt.savefig(f'pic/{file_name}_{cmap}.jpg', bbox_inches='tight', dpi=300)
    plt.close()


def plot_demo_timestamps(tri_to_time, ticks, fs1, fs2, file_name='demo_time'):
    fig = plt.figure(figsize=(7.5, 6))
    for tri, times in tri_to_time.items():
        s = tri[0]
        point_size = 80
        plt.scatter(times, [s] * len(times), s=point_size)
    plt.tick_params(labelsize=fs2)
    # plt.rcParams['ytick.major.width'] = 4.0
    # ax.set_yticks([0, 1, 2, 3, 4], ticks)
    # ax.set_xlabel('Timestamps')
    # ax.set_ylabel('Personal Decision')plt.rcParams['xtick.labelsize'] = 16
    plt.axvspan(-5, 75, facecolor='#C0C0C0', alpha=0.5)
    plt.axvspan(150, 250, facecolor='#C0C0C0', alpha=0.5)
    
    plt.yticks([0, 1, 2, 3, 4], ticks, fontsize=fs2, fontweight='bold')
    plt.xlabel('Timestamps', fontsize=fs2, fontweight='bold')
    # ax.tick_params(labelsize=fs2, fontweight='bold')
    plt.savefig(f'pic/{file_name}.jpg', bbox_inches='tight', dpi=300)
    plt.close()