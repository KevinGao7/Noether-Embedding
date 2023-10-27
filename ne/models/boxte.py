from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

import math
import torch
import numpy as np

from .base import BaselineModel
    

class BaseBoxE(nn.Module):
    """
    Base class for BoxTE, BoxE, TBoxE, and DEBoxE
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super(BaseBoxE, self).__init__()
        self.init_f = torch.nn.init.uniform_
        if norm_embeddings:
            self.embedding_norm_fn = nn.Tanh()
        else:
            self.embedding_norm_fn = nn.Identity()
        self.device = device
        self.embedding_dim = embedding_dim
        self.relation_id_offset = relation_ids[0]
        self.timestamps = timestamps
        self.max_time = max(timestamps) + 1
        self.nb_relations = len(relation_ids)
        self.nb_entities = len(entity_ids)
        # assert sorted(
        #     entity_ids + relation_ids) == entity_ids + relation_ids, 'ids need to be ascending ints from 0 with entities coming before relations'

        self.r_head_base_points = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_head_widths = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_head_size_scales = nn.Embedding(self.nb_relations, 1)
        self.r_tail_base_points = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_tail_widths = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_tail_size_scales = nn.Embedding(self.nb_relations, 1)

        self.entity_bases = nn.Embedding(self.nb_entities, embedding_dim)
        self.entity_bumps = nn.Embedding(self.nb_entities, embedding_dim)

        self.init_f(self.r_head_base_points.weight, *weight_init_args)
        self.init_f(self.r_head_size_scales.weight, -1, 1)
        self.init_f(self.r_head_widths.weight, *weight_init_args)
        self.init_f(self.r_tail_base_points.weight, *weight_init_args)
        self.init_f(self.r_tail_size_scales.weight, -1, 1)
        self.init_f(self.r_tail_widths.weight, *weight_init_args)
        self.init_f(self.entity_bases.weight, *weight_init_args)
        self.init_f(self.entity_bumps.weight, *weight_init_args)

    def shape_norm(self, t, dim):
        # taken from original BoxE implementation (https://github.com/ralphabb/BoxE)
        step1_tensor = torch.abs(t)
        step2_tensor = step1_tensor + (10 ** -8)
        log_norm_tensor = torch.log(step2_tensor)
        step3_tensor = torch.mean(log_norm_tensor, dim=dim, keepdim=True)

        norm_volume = torch.exp(step3_tensor)
        return t / norm_volume

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def get_r_idx_by_id(self, r_ids):
        """@:param r_names tensor of realtion ids"""
        return r_ids - self.relation_id_offset

    def get_e_idx_by_id(self, e_ids):
        return e_ids

    def compute_relation_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        # get relevant embeddings
        r_head_bases = self.r_head_base_points(rel_idx)
        r_tail_bases = self.r_tail_base_points(rel_idx)

        r_head_widths = self.shape_norm(self.r_head_widths(rel_idx), dim=2)  # normalize relative widths
        r_tail_widths = self.shape_norm(self.r_tail_widths(rel_idx), dim=2)

        r_head_scales = nn.functional.elu(self.r_head_size_scales(rel_idx)) + 1  # ensure scales > 0
        r_tail_scales = nn.functional.elu(self.r_tail_size_scales(rel_idx)) + 1
        # compute scaled widths
        head_deltas = torch.multiply(r_head_widths, r_head_scales)
        tail_deltas = torch.multiply(r_tail_widths, r_tail_scales)
        # compute corners from base and width
        head_corner_1 = r_head_bases + head_deltas
        head_corner_2 = r_head_bases - head_deltas
        tail_corner_1 = r_tail_bases + tail_deltas
        tail_corner_2 = r_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        r_head_boxes = torch.stack((head_upper, head_lower), dim=2)
        r_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2)
        return self.embedding_norm_fn(torch.stack((r_head_boxes, r_tail_boxes), dim=2))

    def compute_entity_embeddings(self, tuples):
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        return self.embedding_norm_fn(torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2))

    def compute_embeddings(self, tuples):
        # get relevant entity bases and bumps
        return self.compute_entity_embeddings(tuples), self.compute_relation_embeddings(tuples)

    def forward_negatives(self, negatives):
        """
        @:return tuple (entities, relations, times) containing embeddings with
            entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
            relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
            times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        """
        return self.compute_embeddings(negatives)

    def forward_positives(self, positives):
        return self.compute_embeddings(positives)

    def forward(self, positives, negatives):
        '''
        @:param positives tensor containing id's for entities, relations and times of shape (1, 4, batch_size)
            and where dim 1 indicates 0 -> head, 1 -> relation, 2 -> tail, 3 -> time
        @:param negatives tensor containing id's for entities, relations and times of shape (nb_negative_samples, 4, batch_size)
            and where dim 1 indicates 0 -> head, 1 -> relation, 2 -> tail, 3 -> time
        @:return tuple ((p_entities, p_relations, p_times), (n_entities, n_relations, n_times)) with
            p_entities.shape = (1, batch_size, arity, embedding_dim)
            p_relations.shape = (1, batch_size, arity, 2, embedding_dim)
            p_times.shape = (1, batch_size, arity, 2, embedding_dim)
            n_entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
            n_relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
            n_times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        '''
        positive_emb = self.forward_positives(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb


def to_cartesian(vecs, device):
    nb_examples, batch_size, nb_timebumps, emb_dim = vecs.shape
    og_shape = (nb_examples, batch_size, nb_timebumps, emb_dim)
    flat_shape = (nb_examples * batch_size * nb_timebumps, emb_dim)
    vecs = vecs.view(flat_shape)

    r = vecs[:, 0]
    angles = vecs[:, 1:]
    cos_vec = angles.cos()
    sin_vec = angles.sin()
    xs = []
    running_sin = torch.ones(len(vecs), device=device)
    for i_a, a in enumerate(angles.t()):  # iterate over embedding_dim-1
        xs.append(r * running_sin * cos_vec[:, i_a])
        running_sin = running_sin * sin_vec[:, i_a].clone()
    xs.append(r * running_sin)
    return torch.stack(xs, dim=1).view(og_shape)


def to_angle_interval(angles):
    '''maps angles to [0, 2*pi) interval '''
    angles_by_twopi = angles/(2*math.pi)
    return (angles_by_twopi - torch.floor(angles_by_twopi)) * 2 * math.pi


class BoxTE_Original(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False, time_weight=1, use_r_factor=False, use_e_factor=False,
                 nb_timebumps=1, use_r_rotation=False, use_e_rotation=False, nb_time_basis_vecs=-1,
                 norm_time_basis_vecs=False, use_r_t_factor=False, dropout_p=0.0, arity_spec_timebumps=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        self.nb_time_basis_vecs = nb_time_basis_vecs
        self.norm_time_basis_vecs = norm_time_basis_vecs
        self.use_r_factor, self.use_e_factor, self.use_r_t_factor = use_r_factor, use_e_factor, use_r_t_factor
        self.use_r_rotation, self.use_e_rotation = use_r_rotation, use_e_rotation
        self.nb_timebumps = nb_timebumps
        self.time_weight = time_weight
        self.droput_p = dropout_p
        self.droput = nn.Dropout(dropout_p)
        self.arity_spec_timebumps = arity_spec_timebumps
        if not self.nb_time_basis_vecs > 0:  # don't factorize time bumps, learn them directly/explicitly
            self.factorize_time = False
            if self.arity_spec_timebumps:
                self.head_time_bumps = nn.Parameter(torch.empty(self.max_time, self.nb_timebumps, self.embedding_dim))
                self.tail_time_bumps = nn.Parameter(torch.empty(self.max_time, self.nb_timebumps, self.embedding_dim))
                self.init_f(self.head_time_bumps, *weight_init_args)
                self.init_f(self.tail_time_bumps, *weight_init_args)
            else:
                self.time_bumps = nn.Parameter(torch.empty(self.max_time, self.nb_timebumps, self.embedding_dim))
                self.init_f(self.time_bumps, *weight_init_args)
        else:  # factorize time bumps into two tensors
            self.factorize_time = True
            if self.arity_spec_timebumps:
                self.head_time_bumps_a = nn.Parameter(torch.empty(self.nb_timebumps, self.max_time, self.nb_time_basis_vecs))
                self.head_time_bumps_b = nn.Parameter(
                    torch.empty(self.nb_timebumps, self.nb_time_basis_vecs, self.embedding_dim))
                self.tail_time_bumps_a = nn.Parameter(
                    torch.empty(self.nb_timebumps, self.max_time, self.nb_time_basis_vecs))
                self.tail_time_bumps_b = nn.Parameter(
                    torch.empty(self.nb_timebumps, self.nb_time_basis_vecs, self.embedding_dim))
                self.init_f(self.head_time_bumps_a, *weight_init_args)
                self.init_f(self.head_time_bumps_b, *weight_init_args)
                self.init_f(self.tail_time_bumps_a, *weight_init_args)
                self.init_f(self.tail_time_bumps_b, *weight_init_args)

            else:
                self.time_bumps_a = nn.Parameter(torch.empty(self.nb_timebumps, self.max_time, self.nb_time_basis_vecs))
                self.time_bumps_b = nn.Parameter(torch.empty(self.nb_timebumps, self.nb_time_basis_vecs, self.embedding_dim))
                self.init_f(self.time_bumps_a, *weight_init_args)
                self.init_f(self.time_bumps_b, *weight_init_args)
        if self.use_r_factor:
            if self.arity_spec_timebumps:
                self.head_r_factor = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
                self.tail_r_factor = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
                torch.nn.init.normal_(self.head_r_factor, 1, 0.1)
                torch.nn.init.normal_(self.tail_r_factor, 1, 0.1)
            else:
                self.r_factor = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
                torch.nn.init.normal_(self.r_factor, 1, 0.1)
                self.head_r_factor = self.r_factor
                self.tail_r_factor = self.r_factor
        if self.use_r_t_factor:
            self.r_t_factor = nn.Parameter(torch.empty(self.nb_relations, self.max_time, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.r_t_factor, 1, 0.1)
        if self.use_e_factor:
            self.e_factor = nn.Parameter(torch.empty(self.nb_entities, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.e_factor, 1, 0.1)
        if self.use_r_rotation:
            self.r_angles = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.r_angles, 0, 0.1)
        if self.use_e_rotation:
            self.e_angles = nn.Parameter(torch.empty(self.nb_entities, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.e_angles, 0, 0.1)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        if not self.arity_spec_timebumps:
            if self.use_r_factor:
                self.head_r_factor = self.r_factor
                self.tail_r_factor = self.r_factor
            if not self.factorize_time:
                self.head_time_bumps = self.time_bumps
                self.tail_time_bumps = self.time_bumps
            else:
                self.head_time_bumps_a = self.time_bumps_a
                self.tail_time_bumps_a = self.tail_time_bumps
                self.head_time_bumps_b = self.time_bumps_b
                self.tail_time_bumps_b = self.time_bumps_b

    def dropout_timebump(self, bumps):
        if self.droput_p == 0:
            return bumps
        max_time, nb_timebumps, embedding_dim = bumps.shape
        bump_mask = self.droput(torch.ones(max_time, nb_timebumps, device=self.device))
        if self.training:  # undo dropout scaling, not needed here since we're using a mask
            bump_mask = bump_mask * (1 - self.droput_p)
        bump_mask = bump_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)
        return bumps * bump_mask

    def compute_timebumps(self, is_tail=False, ignore_dropout=False):
        if not self.factorize_time:
            if self.arity_spec_timebumps:
                bumps = self.tail_time_bumps if is_tail else self.head_time_bumps
            else:
                bumps = self.time_bumps
            return bumps if ignore_dropout else self.dropout_timebump(bumps)
        ####
        if self.arity_spec_timebumps:
            bumps_a = self.tail_time_bumps_a if is_tail else self.head_time_bumps_a
            bumps_b = self.tail_time_bumps_b if is_tail else self.tail_time_bumps_b
        else:
            bumps_a = self.time_bumps_a
            bumps_b = self.time_bumps_b
        if self.norm_time_basis_vecs:
            bumps_a = torch.nn.functional.softmax(bumps_a, dim=1)
        bumps = torch.matmul(bumps_a, bumps_b).transpose(0, 1)
        return bumps if ignore_dropout else self.dropout_timebump(bumps)

    def compute_combined_timebumps(self, ignore_dropout=False):
        if not self.arity_spec_timebumps:
            return self.compute_timebumps(ignore_dropout=ignore_dropout)
        else:
            head_bumps = self.compute_timebumps(is_tail=False, ignore_dropout=ignore_dropout)
            tail_bumps = self.compute_timebumps(is_tail=True, ignore_dropout=ignore_dropout)
            return torch.cat((head_bumps, tail_bumps), dim=1)

    def apply_rotation(self, vecs, angles):
        nb_examples, batch_size, nb_timebumps, emb_dim = vecs.shape
        og_shape = (nb_examples, batch_size, nb_timebumps, emb_dim)
        flat_shape = (nb_examples * batch_size * nb_timebumps, emb_dim)
        angles = to_angle_interval(angles)
        angles = torch.cat([angles for _ in range(emb_dim - 1)], dim=3)  # same angle for all dims
        vecs_sph = vecs.view(flat_shape)  # interpret given vectors as spherical coordinates
        vecs_sph[:, 1:] = to_angle_interval(vecs_sph[:, 1:])  # angles need to be in [0, 2pi)
        vecs_sph[:, 1:] += angles.view((nb_examples * batch_size * nb_timebumps, emb_dim-1))  # apply angles
        vecs_sph[:, 1:] = to_angle_interval(vecs_sph[:, 1:])  # angles need to be in [0, 2pi)
        return vecs_sph.view(og_shape).abs()  # radii need to be >= 0

    def index_bumps(self, bumps, idx):
        ''' For atemporal facts, return zero bump; for temporal fact, return appropriate time bump '''
        zeros = torch.zeros(self.nb_timebumps, self.embedding_dim, device=self.device)
        ones = torch.ones(self.nb_timebumps, self.embedding_dim, device=self.device)
        zero_one = torch.stack((zeros, ones))
        mask_idx = torch.where(idx > 0, torch.tensor([1], device=self.device), torch.tensor([0], device=self.device))
        temp_fact_mask = zero_one[mask_idx]
        return bumps[idx] * temp_fact_mask

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        time_idx = tuples[:, 3]
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_bumps_h = self.compute_timebumps(is_tail=False)
        time_bumps_t = self.compute_timebumps(is_tail=True)
        time_vecs_h = self.index_bumps(time_bumps_h, time_idx)
        time_vecs_t = self.index_bumps(time_bumps_t, time_idx)
        if self.use_r_rotation:
            time_vecs_h = self.apply_rotation(time_vecs_h, self.r_angles[rel_idx, :, :])
            time_vecs_t = self.apply_rotation(time_vecs_t, self.r_angles[rel_idx, :, :])
        if self.use_e_rotation:
            time_vecs_h = self.apply_rotation(time_vecs_h, self.e_angles[e_h_idx, :, :])
            time_vecs_t = self.apply_rotation(time_vecs_t, self.e_angles[e_t_idx, :, :])
        if self.use_r_rotation or self.use_e_rotation:
            # if rotations are used, we interpret saved time bumps as spherical coordinates
            # so we need to transform to cartesian before applying the bumps
            time_vecs_h = to_cartesian(time_vecs_h, device=self.device)
            time_vecs_t = to_cartesian(time_vecs_t, device=self.device)
        if self.use_r_factor:
            time_vecs_h *= self.head_r_factor[rel_idx, :, :]
            time_vecs_t *= self.tail_r_factor[rel_idx, :, :]
        if self.use_e_factor:
            time_vecs_h *= self.e_factor[e_h_idx, :, :]
            time_vecs_t *= self.e_factor[e_t_idx, :, :]
        if self.use_r_t_factor:
            time_vecs_h *= self.r_t_factor[rel_idx, time_idx, :, :]
            time_vecs_t *= self.r_t_factor[rel_idx, time_idx, :, :]
        time_vecs_h, time_vecs_t = time_vecs_h.sum(dim=2), time_vecs_t.sum(dim=2)  # sum over all time bumps
        time_vecs = torch.stack((time_vecs_h, time_vecs_t), dim=2)  # apply to both heads and tails
        entity_embs = entity_embs + self.time_weight * time_vecs
        return self.embedding_norm_fn_(entity_embs), self.embedding_norm_fn_(relation_embs), None


class BoxELoss():
    """
    Callable that will either perform uniform, self-adversarial, or cross entroy loss, depending on the setting in @:param args
    """
    def __init__(self, args, device='cpu', timebump_shape=None):
        self.use_time_reg = args.use_time_reg
        self.use_ball_reg = args.use_ball_reg
        self.time_reg_weight = args.time_reg_weight
        self.ball_reg_weight = args.ball_reg_weight
        self.time_reg_order = args.time_reg_order
        self.ball_reg_order = args.ball_reg_order
        if args.loss_type in ['uniform', 'u']:
            self.loss_fn = uniform_loss
            self.fn_kwargs = {'gamma': args.margin, 'w': 1.0 / args.num_negative_samples}
        elif args.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
            self.loss_fn = adversarial_loss
            self.fn_kwargs = {'gamma': args.margin, 'alpha': args.adversarial_temp, 'device': device}
        elif args.loss_type in ['cross entropy', 'cross-entropy', 'ce']:
            self.loss_fn = cross_entropy_loss
            self.fn_kwargs = {'ce_loss': torch.nn.CrossEntropyLoss(reduction=args.ce_reduction),
                              'device': device}
        if self.use_time_reg:
            if timebump_shape is None:
                raise ValueError('Time reg is enabled but timebump shape is not provided.')
            self.diff_matrix = make_diff_matrix(timebump_shape, device=device)

    def __call__(self, positive_tuples, negative_tuples, time_bumps=None):
        l = self.loss_fn(positive_tuples, negative_tuples, **self.fn_kwargs)
        if self.use_time_reg:
            l = l + self.time_reg_weight * self.time_reg(time_bumps, norm_ord=self.time_reg_order)
        if self.use_ball_reg:
            l = l + self.ball_reg_weight * self.ball_reg(entities=positive_tuples[0], relations=positive_tuples[1], norm_ord=self.ball_reg_order)
        return l

    def time_reg(self, time_bumps, norm_ord=4):
        """Temporal smoothness regulariser from the paper'Tensor Decomposition for Temporal Knowledge Base Completion',
        Lacroix et. al."""
        # max_time, nb_timebumps, embedding_dim = time_bumps.shape
        time_bumps = time_bumps.transpose(0, 1)
        diffs = self.diff_matrix.matmul(time_bumps)
        return (torch.linalg.norm(diffs, ord=norm_ord, dim=2) ** norm_ord).mean()

    def ball_reg(self, entities, relations, norm_ord=4):
        """Regulariser inspired by the paper 'ChronoR: Rotation Based Temporal Knowledge Graph Embedding',
        Sadeghian et. al."""
        heads = entities[:, :, 0, :]
        tails = entities[:, :, 1, :]
        box_centers = (relations[:, :, :, 0, :] + relations[:, :, :, 1, :]) / 2
        head_centers = box_centers[:, :, 0, :]
        tail_centers = box_centers[:, :, 1, :]
        return (torch.linalg.norm(heads, ord=norm_ord, dim=-1) ** norm_ord
                + torch.linalg.norm(tails, ord=norm_ord, dim=-1) ** norm_ord
                + torch.linalg.norm(head_centers, ord=norm_ord, dim=-1) ** norm_ord
                + torch.linalg.norm(tail_centers, ord=norm_ord, dim=-1) ** norm_ord).mean()


def make_diff_matrix(timebump_shape, device):
    (max_time, nb_timebumps, embedding_dim) = timebump_shape
    m = torch.eye(max_time, max_time, requires_grad=False, device=device)
    for i in range(m.shape[0] - 1):
        m[i, i + 1] = -1
    m = m[:-1, :]
    return m.unsqueeze(0)


def dist(entity_emb, boxes):
    """
     assumes box is tensor of shape (nb_examples, batch_size, arity, 2, embedding_dim)
     nb_examples is relevant for negative samples; for positive examples it is 1
     so it contains multiple boxes, where each box has lower and upper boundaries in embedding_dim dimensions
     e.g box[0, n, 0, :] is the lower boundary of the n-th box
     entities are of shape (nb_examples, batch_size, arity, embedding_dim)
    """

    ub = boxes[:, :, :, 0, :]  # upper boundaries
    lb = boxes[:, :, :, 1, :]  # lower boundaries
    c = (lb + ub) / 2  # centres
    w = ub - lb + 1  # widths
    k = 0.5 * (w - 1) * (w - (1 / w))
    d = torch.where(torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub)),
                    torch.abs(entity_emb - c) / w,
                    torch.abs(entity_emb - c) * w - k)
    return d


def score(entities, relations, times, order=2, time_weight=0.5):
    d_r = dist(entities, relations).norm(dim=3, p=order).sum(dim=2)
    if times is not None:
        d_t = dist(entities, times).norm(dim=3, p=order).sum(dim=2)
        return time_weight * d_t + (1 - time_weight) * d_r
    else:
        return d_r


def uniform_loss(positives, negatives, gamma, w):
    """
    Calculates uniform negative sampling loss as presented in RotatE, Sun et. al.
    @:param positives tuple (entities, relations, times), for details see return of model.forward
    @:param negatives tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param w hyperparameter, corresponds to 1/k in RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    eps = torch.finfo(torch.float32).tiny
    s1 = - torch.log(torch.sigmoid(gamma - score(*positives)) + eps)
    s2 = torch.sum(w * torch.log(torch.sigmoid(score(*negatives) - gamma) + eps), dim=0)
    return torch.sum(s1 - s2)


def triple_probs(negative_triples, alpha, device='cpu'):
    eps = torch.finfo(torch.float32).eps
    pre_exp_scores = ((1 / (score(*negative_triples) + eps)) * alpha)
    pre_exp_scores = torch.minimum(pre_exp_scores, torch.tensor([85.0], device=device))  # avoid exp exploding to inf
    scores = pre_exp_scores.exp()
    div = scores.sum(dim=0) + eps
    return scores / div


def adversarial_loss(positive_triple, negative_triples, gamma, alpha, device='cpu'):
    """
    Calculates self-adversarial negative sampling loss as presented in RotatE, Sun et. al.
    @:param positive_triple tuple (entities, relations, times), for details see return of model.forward
    @:param negative_triple tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param alpha hyperparameter, see RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    triple_weights = triple_probs(negative_triples, alpha, device)
    return uniform_loss(positive_triple, negative_triples, gamma, triple_weights)


def cross_entropy_loss(positive_triple, negative_triples, ce_loss, device='cpu'):
    pos_scores = score(*positive_triple)
    neg_scores = score(*negative_triples)
    combined_inv_scores = torch.cat((-pos_scores, -neg_scores), dim=0).t()
    target = torch.zeros((combined_inv_scores.shape[0]), dtype=torch.long, device=device)
    return ce_loss(combined_inv_scores, target)


class BoxTE(BoxTE_Original, BaselineModel):
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        uniform_init_args = [(-0.5/np.sqrt(cfg.embedding_dim)) * cfg.weight_init_factor,
                            0.5/np.sqrt(cfg.embedding_dim) * cfg.weight_init_factor]
        BoxTE_Original.__init__(self, cfg.embedding_dim, cfg.relation_ids, cfg.entity_ids, cfg.timestamps,
                      weight_init_args=uniform_init_args, time_weight=cfg.time_weight,
                      norm_embeddings=cfg.norm_embeddings, use_r_factor=cfg.use_r_factor,
                      use_e_factor=cfg.use_e_factor, nb_timebumps=cfg.nb_timebumps,
                      use_r_rotation=cfg.use_r_rotation, use_e_rotation=cfg.use_e_rotation,
                      nb_time_basis_vecs=cfg.nb_time_basis_vecs,
                      norm_time_basis_vecs=cfg.norm_time_basis_vecs, use_r_t_factor=cfg.use_r_t_factor,
                      dropout_p=cfg.timebump_dropout_p, arity_spec_timebumps=cfg.arity_spec_timebumps)
        
    def forward(self, x):
        batch_size, time_len = x.shape[0], self.cfg.sizes[3]
        times = torch.arange(0, time_len, device=x.device, dtype=torch.long).repeat(x.shape[0]).unsqueeze(1)
        x = x.repeat_interleave(time_len, 0)
        x = torch.cat((x, times), 1)
        x = x.unsqueeze(2)
        emb_tuples = self.forward_positives(x)
        score_ = score(*emb_tuples).view(batch_size, time_len)
        score_ = self.norm_event_score(score_)
        return score_

    def forward_(self, positives, negatives):
        return BoxTE_Original.forward(self, positives, negatives)

    def get_loss_func(self):
        return BoxELoss(self.cfg, device=self.device, timebump_shape=self.compute_combined_timebumps().shape)
