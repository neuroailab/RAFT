# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pdb
import os
import h5py
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from detectron2.modeling import build_backbone
from detectron2.config import get_cfg
from detectron2.layers import Conv2d, get_norm
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.projects.panoptic_deeplab.propagation import GraphPropagation, compute_gt_affinity
from detectron2.projects.panoptic_deeplab.connected_component import label_connected_component
from detectron2.projects.panoptic_deeplab.competition import Competition
from detectron2.projects.panoptic_deeplab.aggregate_pseudolabels import aggregate_multiple_labels
from detectron2.projects.panoptic_deeplab.segmentation_metrics import SegmentationMetrics
from detectron2.projects.spatial_temporal.models.measure_static_seg import (
    static_propagation,
    measure_static_segmentation_metric
)
#from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
#from detectron2.projects.spatial_temporal import add_spatial_temporal_config
import detectron2.projects.spatial_temporal.models.utils as utils
from detectron2.projects.spatial_temporal.models.resnet import ResNetLayer, ResNetBasicBlock, ResNetBottleNeckBlock
import copy
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.projects.spatial_temporal import add_spatial_temporal_config
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GraphStructureInference(nn.Module):
    def __init__(self, cfg, input_dim):
        super(GraphStructureInference, self).__init__()

        # set attributes based on config
        for k, v in cfg.MODEL.GRAPH_INFERENCE_HEAD.items():
            setattr(self, k.lower(), v)
        self.build_buffers()
        self.sample_affinity = None  # determined at runtime

        # check attributes
        assert len(self.full_affinity) <= len(self.levels), (self.full_affinity, self.levels)
        assert len(self.obj_sup) <= 1, self.obj_sup
        if len(self.obj_sup) == 1:
            assert self.obj_sup[0] in ['gt_moving', 'raft_flow', 'flow', 'delta_image', 'delta_image_floodfill', 'object_segments']

        # model parameters
        self.conv = Conv2d(input_dim, self.kq_dim, kernel_size=self.kernel_size, bias=True)
        # self.key = Conv2d(self.kq_dim, self.kq_dim, kernel_size=self.kernel_size, bias=True, padding='same')
        # self.query = Conv2d(self.kq_dim, self.kq_dim, kernel_size=self.kernel_size, bias=True, padding='same')

        if self.linear_kq_proj:
            self.key = nn.Linear(self.kq_dim, self.kq_dim)
            self.query = nn.Linear(self.kq_dim, self.kq_dim)
        else:
            default_level_params = {
                "out_channels": 64,
                "downsampling": 1,
                "block": ResNetBasicBlock,
                "n": 1,
                'use_batchnorm': True
            }

            print('Warning: change key and query projector')
            def _build_level(in_chs):
                return nn.Sequential(
                    ResNetLayer(in_chs, **default_level_params),
                    Conv2d(default_level_params['out_channels'], self.kq_dim, kernel_size=self.kernel_size, bias=True, padding='same'))

            self.key = _build_level(self.kq_dim)
            self.query = _build_level(self.kq_dim)

        if 'playroom' in cfg.DATASETS.TRAIN[0] or 'robonet' in cfg.DATASETS.TRAIN[0]:
            self.size_dict = {2: [128, 128], 3: [64, 64], 4: [32, 32]}
        elif 'bcz' in cfg.DATASETS.TRAIN[0]:
            self.size_dict = {2: [128, 160], 3: [64, 80], 4: [32, 40]}
        elif 'dsr' in cfg.DATASETS.TRAIN[0]:
            self.size_dict = {2: [120, 120], 3: [60, 60], 4: [30, 30]}

        print('Size dict: ', self.size_dict)
        # --- arguments for reproducing the new decoder
        if self.build_feature_pyramid:
            self.kernel_size = 3
            self.in_channels = input_dim
            self.attention_dim = self.kq_dim
            self.attention_kernel_size = 3
            self._compute_values = False
            self.num_levels = 3
            self._set_level_params()
            self._build_levels()

            if self.concat_feature_level is False:
                self.attention_input_channels = [512, 1024, 2048]
            self._build_local_attention_layers()

    def build_buffers(self):

        H, W = self.input_res
        # multi-level input shape
        self.levels = range(self.min_level, self.max_level+1)
        self.input_res = []
        self.stride = {}
        for idx, level in enumerate(self.levels):
            stride = 2 ** idx  # 1, 2, 4, ...
            self.stride[level] = stride
            Hd, Wd = int(H/stride), int(W/stride)
            self.input_res.append([Hd, Wd])

        # create position encoding for computing position affinities
        if self.add_pos_encoding:
            pos_encoding = utils._relative_sine_pos_encoding(
                query_size=[H, W],
                key_size=[H, W],
                dim=self.kq_dim
            )  # [W, W, D], [H, H, D]

            for level in self.levels:
                stride = self.stride[level]
                rel_w_embed = utils.downscale_tensor(pos_encoding[0].unsqueeze(0), stride)[0]
                rel_h_embed = utils.downscale_tensor(pos_encoding[1].unsqueeze(0), stride)[0]
                self.register_buffer('rel_w_embed_%d' % level, rel_w_embed)
                self.register_buffer('rel_h_embed_%d' % level, rel_h_embed)

        # Create local indices buffer for affinity sampling ("local_global_mem_efficient")
        ksize = self.subsample_local_ksize

        if self.inference_subsample_method == 'local_global_mem_efficient':
            for size, level in zip(self.input_res, self.levels):
                self.register_buffer('local_indices_%d' % level,
                                     utils.create_local_ind_buffer(size, ksize, padding=self.sample_inds_padding))
        else:
            raise ValueError(self.inference_subsample_method)

    def forward(self, feats, batched_inputs, sample_inds=None, activated=None):
        """ build outputs at multiple levels"""
        # feature projection
        if list(feats.shape[-2:]) != self.input_res[0]:
            self.adjust_resolution = True
        else:
            self.adjust_resolution = False

        if self.build_feature_pyramid:
            if isinstance(feats, dict):
                assert self.concat_feature_level is False and self.apply_aspp is False
                self.feature_pyramid = [[feats['res%d' % i]]*3 for i in [2, 3, 4]]
            else:
                self.feature_pyramid = self._compute_feature_pyramid(feats)
            keys = queries = None
        else:
            if isinstance(self.key, nn.Linear):
                assert isinstance(self.query, nn.Linear)
                feats = self.conv(feats).permute(0, 2, 3, 1)  # [B, H, W, C]

                # key & query projection
                keys = self.key(feats).permute(0, 3, 1, 2)              # [B, C, H, W]
                queries = self.query(feats).permute(0, 3, 1, 2)         # [B, C, H, W]
            else:
                feats = self.conv(feats)  # [B, C, H, W]
                # key & query projection
                keys = self.key(feats) # [B, C, H, W]
                queries = self.query(feats) # [B, C, H, W]

        out_dict = {}
        self.sample_affinity = self.subsample_affinity and (not (self.eval_full_affinity and (not self.training)))

        # compute affinity and its loss at different resolution
        for stride, level in zip(self.stride, self.levels):
            out_dict[str(level)] = self._forward(level, keys, queries, batched_inputs, sample_inds, activated)

        # sum the affinity losses from each level
        if 'loss' in out_dict[str(level)].keys():
            out_dict['sup_loss'] = sum([out_dict[str(level)]['loss'] for level in self.levels])

        if 'boostrap_loss' in out_dict[str(level)].keys():
            out_dict['boostrap_loss'] = sum([out_dict[str(level)]['boostrap_loss'] for level in self.levels])

        if self.combine_attention:
            out_dict['combine_logits'] = self.combine_multi_level_affinities(out_dict)

        return out_dict

    def _forward(self, level, key, query, batched_inputs, sample_inds, activated):
        """ build outputs at a single level """

        # downscale the key and query tensors according to 'level'
        if not self.build_feature_pyramid:
            stride = self.stride[level]
            key = utils.downscale_tensor(key, stride)            # [B, C, H/stride, W/stride]
            query = utils.downscale_tensor(query, stride)         # [B, C, H/stride, W/stride]
        else:
            key, query, _ = self.feature_pyramid[level - 2]
            level_idx = level - 2
            keys_conv, queries_conv, _ = self.attention_layers[(3*level_idx):(3*level_idx)+3]
            key = keys_conv(key)
            query = queries_conv(query)

        B, C, H, W = key.shape

        # sample indices for inference/propagation
        if sample_inds is None or level > self.min_level:
            sample_inds = self.affinity_sample_indices(level, [B, H, W]) if self.sample_affinity else None

        # compute affinity logits (feature affinity + positional affinity)
        logits_dict = self.feature_affinity_logits(level, key, query, sample_inds)

        if self.add_pos_encoding:
            logits_dict = self.position_affinity_logits(level, logits_dict, query, sample_inds)
        else:
            logits_key = 'subsample_logits' if self.sample_affinity else 'logits'
            logits_dict[logits_key] *= C ** -0.5


        if self.symmetric_affinity:
            logits_dict = self.compute_symmetric_affinity(level, key, query, logits_dict, sample_inds)

        out_dict = logits_dict
        out_dict['sample_inds'] = sample_inds

        # compute losses
        losses_dict = self.compute_losses(logits_dict, sample_inds, batched_inputs, size=[H, W], level=level)

        out_dict.update(losses_dict)

        # # run KP + comp
        # if not self.training and level == self.min_level:
        #     logits = logits_dict['subsample_logits']
        #     segment_dict = self.propagation_competition(logits, sample_inds, batched_inputs, [H, W], activated)
        #     out_dict.update(segment_dict)

        return out_dict

    def compute_symmetric_affinity(self, level, key, query, logits_dict, sample_inds):
        if self.sample_affinity:
            b_inds, n_inds, s_inds = torch.split(sample_inds, [1, 1, 1], dim=0)
            trans_sample_inds = torch.cat([b_inds, s_inds, n_inds], axis=0)
            logits_dict_transpose = self.feature_affinity_logits(level, key, query, trans_sample_inds)

            if self.add_pos_encoding:
                logits_dict_transpose = self.position_affinity_logits(level, logits_dict_transpose, query,
                                                                     trans_sample_inds)

            logits_dict['subsample_logits'] = 0.5 * (logits_dict['subsample_logits'] +
                                                     logits_dict_transpose['subsample_logits'])

        else:
            logits_dict['logits'] = (logits_dict['logits'] + logits_dict['logits'].permute(0, 2, 1)) * 0.5

        return logits_dict

    def affinity_sample_indices(self, level, size):
        # local_indices and local_masks below are stored in the buffers
        # so that we don't have to do the same computation at every iteration

        sampling_method = self.inference_subsample_method
        S = self.subsample_num_samples
        K = self.subsample_local_ksize
        B, H, W = size
        N = H * W

        if not self.adjust_resolution:
            assert sampling_method == 'local_global_mem_efficient'  # this strategy cannot guarantee sampling w/o replacement
            local_inds = getattr(self, 'local_indices_%d' % level)  # access local_indices buffer
            local_inds = local_inds.expand(B, -1, -1)  # tile local indices in batch dimension
        else:
            local_inds = utils.create_local_ind_buffer(size[-2:], K, padding=self.sample_inds_padding)#.cuda()
            local_inds = local_inds.expand(B, -1, -1)  # tile local indices in batch dimension

        device = local_inds.device

        if S-K**2 > 0:
            # compute random global indices
            rand_global_inds = torch.randint(H * W, [B, N, S-K**2], device=device)
            sample_inds = torch.cat([local_inds, rand_global_inds], -1)
        else:
            sample_inds = local_inds

        # create gather indices
        sample_inds = sample_inds.reshape([1, B, N, S])
        batch_inds = torch.arange(B, device=device).reshape([1, B, 1, 1]).expand(-1, -1, N, S)
        node_inds = torch.arange(N, device=device).reshape([1, 1, N, 1]).expand(-1, B, -1, S)
        sample_inds = torch.cat([batch_inds, node_inds, sample_inds], 0).long()  # [3, B, N, S]

        return sample_inds

    def feature_affinity_logits(self, level, key, query, sample_inds):

        B, C, H, W = key.shape

        key = key.reshape([B, C, H * W]).permute(0, 2, 1)      # [B, N, C]
        query = query.reshape([B, C, H * W]).permute(0, 2, 1)  # [B, N, C]
        out_dict = dict()

        if self.sample_affinity:
            gathered_query = utils.gather_tensor(query, sample_inds[[0, 1], ...])
            gathered_key = utils.gather_tensor(key, sample_inds[[0, 2], ...])
            out_dict['subsample_logits'] = (gathered_query * gathered_key).sum(-1)  # [B, N, K]

        if level in self.full_affinity or (self.eval_full_affinity and (not self.training)):
            out_dict['logits'] = torch.matmul(query, key.permute(0, 2, 1))

        return out_dict

    def position_affinity_logits(self, level, logits_dict, x, sample_inds):
        assert self.affinity_func == 'dot_product', "positional encoding doesn't support other affinity function"
        B, C, H, W = x.shape
        N = H * W

        if self.pos_encoding_method == 'relative_sine':
            # downscale position encoding
            rel_w_embed = getattr(self, 'rel_w_embed_%d' % level)
            rel_h_embed = getattr(self, 'rel_h_embed_%d' % level)

            if self.adjust_resolution:
                rel_w_embed = F.interpolate(rel_w_embed[None], size=[H, W], mode='nearest')[0]
                rel_h_embed = F.interpolate(rel_h_embed[None], size=[H, W], mode='nearest')[0]

            rel_w_logits = torch.einsum('bdhw,dwm->bhwm', x, rel_w_embed)
            rel_h_logits = torch.einsum('bdhw,dhm->bhwm', x, rel_h_embed)

            rel_w_logits = rel_w_logits.reshape([B, H * W, W])
            rel_h_logits = rel_h_logits.reshape([B, H * W, H])

            if self.sample_affinity:

                # factor the sample indices (last dimension) into h, w indices
                h_inds = sample_inds[2:3] // W  # [1, B, N, S]
                w_inds = sample_inds[2:3] % W   # [1, B, N, S]

                # concat the batch and node indices for gathering
                h_inds = torch.cat([sample_inds[0:2], h_inds], 0)  # [3, B, N, S]
                w_inds = torch.cat([sample_inds[0:2], w_inds], 0)  # [3, B, N, S]

                # gather the relative logtis
                rel_w_logits = utils.gather_tensor(rel_w_logits, w_inds)
                rel_h_logits = utils.gather_tensor(rel_h_logits, h_inds)

                if self.sample_inds_padding == 'constant':  # mask out invalid padding value:
                    invalid_mask = torch.logical_or(sample_inds[-1] == N, sample_inds[1] == N)
                    rel_w_logits[invalid_mask] = 0.
                    rel_h_logits[invalid_mask] = 0.

                logits_dict['subsample_logits'] += (rel_w_logits + rel_h_logits)
                logits_dict['subsample_logits'] *= C ** -0.5
            elif level in self.full_affinity or (self.eval_full_affinity and (not self.training)):

                logits_dict['feature_logits'] = logits_dict['logits'].clone()
                logits_dict['position_logits'] = (
                        rel_w_logits.unsqueeze(2).expand(-1, -1, H, -1) + \
                        rel_h_logits.unsqueeze(3).expand(-1, -1, -1, W)
                ).reshape([B, N, N])

                logits = logits_dict['logits'].reshape([B, N, H, W]) + \
                         rel_w_logits.unsqueeze(2).expand(-1, -1, H, -1) + \
                         rel_h_logits.unsqueeze(3).expand(-1, -1, -1, W)
                logits = logits.reshape([B, N, N])

                logits *= C ** -0.5
                logits_dict['logits'] = logits  # modify logits in-place
            else:
                raise ValueError
        else:
            raise NotImplementedError
        return logits_dict

    def compute_targets(self, batched_inputs, sample_inds, size):

        if len(self.obj_sup) > 0:
            labels = None
            assert len(self.obj_sup) == 1
            key = self.obj_sup[0]
            H, W = size
            B = batched_inputs[key].shape[0]

            # supervise_region = F.interpolate(batched_inputs[key].float(), [H, W], mode='nearest', align_corners=None)
            # supervise_region = supervise_region.sum(1) > 0

            # plt.imshow(supervise_region[0].cpu())
            # plt.show()
            # plt.close()

            # if self.sample_affinity:
            #     # modify this line to account for invalid index in sample inds
            #     # when the samples_inds_padding is 'constant'
            #     if self.sample_inds_padding == 'constant':
            #         targets = utils.gather_tensor(supervise_region.reshape([B, H * W, 1]), sample_inds[[0, 2]], invalid=0)
            #     else:
            #         targets = supervise_region.reshape([B, H * W])[list(sample_inds[[0, 2]])]
            # else:
            #     targets = supervise_region.reshape([B, 1, H * W]).expand(-1, H * W, -1)

            if torch.unique(batched_inputs[key]).shape[0] > 2 or self.obj_sup[0] == 'object_segments':
                # multiple motion segments
                labels = batched_inputs[key]
                if len(labels.shape) == 3:
                    labels = labels.unsqueeze(1)
                labels = F.interpolate(labels.float(), [H, W], mode='nearest', align_corners=None)
                supervise_region = labels > 0
                labels = labels.view(B, H * W, 1)

                if self.sample_affinity:
                    targets = torch.eq(labels, utils.gather_tensor(labels, sample_inds[[0, 2]]).squeeze(-1)).float()  # [B, N, K]
                else:
                    targets = torch.eq(labels.unsqueeze(-1), labels.unsqueeze(-2)).float()  # [B, N, N]
            else:
                # single motion segments
                supervise_region = F.interpolate(batched_inputs[key].float(), [H, W], mode='nearest', align_corners=None)
                supervise_region = supervise_region.sum(1) > 0

                if self.sample_affinity:
                    # modify this line to account for invalid index in sample inds
                    # when the samples_inds_padding is 'constant'
                    if self.sample_inds_padding == 'constant':
                        targets = utils.gather_tensor(supervise_region.reshape([B, H * W, 1]), sample_inds[[0, 2]], invalid=0)
                    else:
                        targets = supervise_region.reshape([B, H * W])[list(sample_inds[[0, 2]])]
                else:
                    targets = supervise_region.reshape([B, 1, H * W]).expand(-1, H * W, -1)
        else:
            print('Warning: using full supervision')
            '''  Full Supervision  '''
            labels = utils._get_gt_labels(batched_inputs['objects'].float(), size)
            B, H, W = labels.shape
            labels = labels.reshape([B, -1])  # [B, N]

            supervise_region = torch.ones([B, H, W]).float().to(labels.device)
            if self.sample_affinity:
                labels = labels.unsqueeze(-1)
                targets = torch.eq(labels, utils.gather_tensor(labels, sample_inds[[0, 2]]).squeeze(-1)).float() # [B, T, K]
            else:
                targets = torch.eq(labels.unsqueeze(-1), labels.unsqueeze(-2)).float()  # [B, T, N]

        return targets, labels, supervise_region

    def compute_losses(self, logits_dict, sample_inds, batched_inputs, size, level):
        """ build loss for a single level """

        logits_key = 'subsample_logits' if self.sample_affinity else 'logits'
        logits = logits_dict[logits_key]

        targets, labels, supervise_region = self.compute_targets(batched_inputs, sample_inds, size)


        loss = utils.kl_divergence(logits, targets, label_smoothing=(self.label_smoothing and self.training))
        mask = supervise_region.reshape(loss.shape)
        loss = (loss * mask).sum() / (mask.sum() + 1e-9)
        # if self.training and self.obj_sup[0] == 'motion_segments':
        #     loss.backward()
        #     pdb.set_trace()
        return {
            'loss': loss,
            'targets': targets,
            'labels': labels,
            'supervise_region': supervise_region,
        }


    def combine_multi_level_affinities(self, out):
        adj_list = []

        for level in range(self.min_level, self.max_level+1):
            adj = out[str(level)]['subsample_logits']
            B, N, K = adj.shape
            H, W = self.size_dict[level]
            assert K == self.subsample_local_ksize ** 2, "current impl only supports local affinities"
            assert N == H * W, (N, H, W)
            adj = adj.permute(0, 2, 1).reshape(B, K, H, W)
            adj = F.interpolate(adj, size=self.size_dict[self.min_level], mode='nearest')
            adj_list.append(adj)

        adj = torch.stack(adj_list, -1).sum(-1)
        adj = adj.flatten(2, 3).permute(0, 2, 1)  # [B, N, K]

        return adj

    def propagation_competition(self, affinities, sample_inds, batched_inputs, size, activated):
        raise NotImplementedError("handle sample indices in constant padding mode")
        if len(affinities.shape) == 3:
            affinities = affinities.unsqueeze(1)

        out = static_propagation(static_affinities=affinities, size=size, tau=1., cc_min=self.cc_min_area, sample_inds=sample_inds, h0=None, activated=activated)
        metric, vis = measure_static_segmentation_metric(out, batched_inputs, size, segment_key='cc_labels')
        return {
            'segment_metric': metric,
            'segment_vis': vis,
            'cc_labels': out['cc_labels']
        }

    def _set_level_params(self):

        default_level_params = {
            "out_channels": 64,
            "downsampling": 1,
            "block": ResNetBasicBlock,
            "n": 1,
            'use_batchnorm': True
        }

        self.level_params = [copy.deepcopy(default_level_params)] * (self.num_levels - 1)

    def _build_levels(self):

        levels = [nn.ModuleList([nn.Identity()] * 3)]

        def _build_level(in_chs, lev_idx):
            return nn.Sequential(
                ResNetLayer(in_chs, **self.level_params[lev_idx]),
                nn.MaxPool2d(2, 2))

        in_channels_now = self.in_channels
        self.attention_input_channels = [self.in_channels]
        for level_idx in range(self.num_levels - 1):

            level = nn.ModuleList([
                _build_level(in_channels_now, level_idx),
                _build_level(in_channels_now, level_idx),
                _build_level(in_channels_now, level_idx) if self._compute_values else nn.Identity()
            ])
            levels.append(level)
            in_channels_now = level[0][0].blocks[-1].out_channels
            self.attention_input_channels.append(in_channels_now)

        self.level_modules = nn.ModuleList(levels)

    def _build_local_attention_layers(self):

        attention_layers = []
        for level_idx in range(self.num_levels):
            key_conv = nn.Conv2d(
                self.attention_input_channels[level_idx],
                self.attention_dim,
                self.attention_kernel_size,
                bias=True, padding='same')
            query_conv = nn.Conv2d(
                self.attention_input_channels[level_idx],
                self.attention_dim,
                self.attention_kernel_size,
                bias=True, padding='same')
            attention_layers.extend([key_conv, query_conv])

            if self._compute_values:
                value_conv = nn.Conv2d(
                    self.attention_input_channels[level_idx],
                    self.K + 1,
                    self.values_kernel_size,
                    bias=True, padding='same')
            else:
                value_conv = nn.Identity()
            attention_layers.append(value_conv)

        self.attention_layers = nn.ModuleList(attention_layers)

    def _compute_feature_pyramid(self, x):

        keys_inp, queries_inp, values_inp = x, x, None
        level_outputs = [(keys_inp, queries_inp, values_inp)]
        self.level_sizes = [list(keys_inp.shape[-2:])]
        self.level_num_nodes = [keys_inp.size(-2) * keys_inp.size(-1)]

        for level_idx, level in enumerate(self.level_modules[1:]):
            k, q, v = level_outputs[-1]
            level_k, level_q, level_v = level
            level_out = (level_k(k), level_q(q), level_v(v) if v is not None else None)
            level_outputs.append(level_out)
            self.level_sizes.append(list(level_out[0].shape[-2:]))
            self.level_num_nodes.append(level_out[0].size(-2) * level_out[0].size(-1))

        return level_outputs



class Encoder(DeepLabV3PlusHead):
    @configurable
    def __init__(self, cfg, **kwargs):
        input_shape = {'res2': ShapeSpec(channels=256, height=None, width=None, stride=4),
                       'res3': ShapeSpec(channels=512, height=None, width=None, stride=8),
                       'res5': ShapeSpec(channels=2048, height=None, width=None, stride=16)}
        self.apply_aspp = cfg.MODEL.GRAPH_INFERENCE_HEAD.APPLY_ASPP

        super().__init__(input_shape, not_use=(not self.apply_aspp), **kwargs)

        self.backbone = build_backbone(cfg)
        self.min_level = cfg.MODEL.GRAPH_INFERENCE_HEAD.MIN_LEVEL
        self.max_level = cfg.MODEL.GRAPH_INFERENCE_HEAD.MAX_LEVEL
        self.concat_feature_level = cfg.MODEL.GRAPH_INFERENCE_HEAD.CONCAT_FEATURE_LEVEL
        self.concat_bilinear = cfg.MODEL.GRAPH_INFERENCE_HEAD.CONCAT_BILINEAR
        self.kq_dim = cfg.MODEL.GRAPH_INFERENCE_HEAD.KQ_DIM
        self.levels = list(range(self.min_level, self.max_level+1))

        channels = {2: 256, 3: 512, 4: 1024}
        # lateral conv for multi-level features concatenation
        for level in self.levels:
            setattr(self, 'lateral_conv_%d' % level, Conv2d(channels[level], 512, kernel_size=1, bias=True))


    @classmethod
    def from_config(cls, cfg):

        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.GRAPH_INFERENCE_HEAD.CONVS_DIM] * (
                len(cfg.MODEL.GRAPH_INFERENCE_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.GRAPH_INFERENCE_HEAD.ASPP_CHANNELS]

        ret = dict(
            cfg=cfg,
            project_channels=cfg.MODEL.GRAPH_INFERENCE_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.GRAPH_INFERENCE_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.GRAPH_INFERENCE_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.GRAPH_INFERENCE_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.GRAPH_INFERENCE_HEAD.NORM,
            train_size=train_size,
        )
        return ret

    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)

        if self.apply_aspp:
            features = super().layers(features)
            return features
        elif self.concat_feature_level:
            features = self.concat_multi_level(features)  # construct feature pyramid instead
            return features[self.min_level]
        else:
            return features

        # else:
        #     return features['res%d' % self.min_level]


    def concat_multi_level(self, features):

        feats_lateral = {l: getattr(self, 'lateral_conv_%d' % l)(features['res%d' % l]) for l in self.levels}

        # Adds top-down path.
        backbone_max_level = self.levels[-1]
        feats = {backbone_max_level: feats_lateral[backbone_max_level]}

        for level in range(backbone_max_level - 1, self.min_level - 1, -1):
            feats[level] = torch.cat([
                F.interpolate(feats[level + 1], scale_factor=2.0,
                              mode='bilinear' if self.concat_bilinear else 'nearest'),
                feats_lateral[level]
            ], 1)

        return feats


class EISEN(nn.Module):
    def __init__(self, args):
        super(EISEN, self).__init__()
        cfg = get_cfg()
        add_spatial_temporal_config(cfg)
        cfg.merge_from_file('./core/RAFT_OED_128.yaml')
        self.encoder = Encoder(cfg)
        self.affinity_decoder = GraphStructureInference(cfg, input_dim=128)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image, motion_segments):
        features = self.encoder(image)
        inputs = {'objects': motion_segments}
        output = self.affinity_decoder(features, inputs)

        return output


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_spatial_temporal_config(cfg)
    config_file = '../../configs/TDW-Playroom-PanopticSegmentation/graph.yaml'
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    net = GraphStructureInference(cfg, input_dim=512).cuda()
    encoder = Encoder(cfg).cuda()
    batched_inputs = torch.load('../../notebook/saved_inputs.pt')
    features = encoder(None, batched_inputs)
    net(features, batched_inputs)
