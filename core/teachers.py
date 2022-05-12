from functools import partial
from pathlib import Path
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from raft import (RAFT,
                  CentroidRegressor,
                  ThingsClassifier,
                  MotionClassifier,
                  BoundaryClassifier)

from eisen import EISEN
from core.utils.utils import softmax_max_norm

import dorsalventral.models.layers as layers
import dorsalventral.models.targets as targets
import dorsalventral.models.fire_propagation as fprop
import dorsalventral.models.segmentation.competition as competition

import sys

CentroidMaskTarget = partial(targets.CentroidTarget, normalize=True, return_masks=True)
ForegroundMaskTarget = partial(targets.MotionForegroundTarget, resolution=8, num_masks=32)
IsMovingTarget = partial(targets.IsMovingTarget, warp_radius=3)
DiffusionTarget = partial(fprop.DiffusionTarget,
                          warp_radius=3,
                          boundary_radius=5)

def get_args(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--motion_path', type=str,
        default='../checkpoints/15000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall-ctd.pth')
    parser.add_argument(
        '--boundary_path', type=str,
        default='../checkpoints/40000_boundaryStaticReg-rnd0-tdw-bs4-large-dtarg-nthr0-cthr075-pr1-tds2-fullplayall.pth')
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--static_coords', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd)
    return args

def path_to_args(path):
    args = {
        'small': 'small' in path,
        'gate_stride': 2 if ('gs1' not in path) else 1
    }
    if 'boundary' in path:
        args.update({
            'orientation_type': 'regression' if ('Reg' in path) else 'classification',
            'static_input': 'Static' in path
        })
    return args

def load_model(load_path,
               model_class=None,
               parse_path=False,
               ignore_prefix=None,
               **kwargs):

    path = Path(load_path) if load_path else None

    def _get_model_class(name):
        cls = None
        if 'raft' in name:
            cls = RAFT
        elif 'thing' in name:
            cls = ThingsClassifier
        elif 'centroid' in name:
            cls = CentroidRegressor
        elif 'motion' in name:
            cls = MotionClassifier
        elif 'boundary' in name:
            cls = BoundaryClassifier
        elif 'flow' in name:
            cls = RAFT
        elif 'eisen' in name:
            cls = EISEN
        else:
            raise ValueError("Couldn't identify a model class associated with %s" % name)
        return cls

    if model_class is None:
        cls = _get_model_class(path.name)
    else:
        cls = _get_model_class(model_class)
    assert cls is not None, "Wasn't able to infer model class"

    ## get the args
    if parse_path:
        kwargs.update(path_to_args(load_path))
    args = get_args("")
    for k,v in kwargs.items():
        args.__setattr__(k,v)

    # build model
    if cls.__name__ == 'EISEN':
        print("LOAD EISEN")
        model = cls(args, **kwargs)
    else:
        model = cls(args)
    if load_path is not None:
        weight_dict = torch.load(load_path)
        new_dict = dict()
        for k in weight_dict.keys():
            if 'module' in k:
                new_dict[k.split('module.')[-1]] = weight_dict[k]

        if ignore_prefix is not None:
            new_dict_1 = dict()
            for k, v in new_dict.items():
                new_dict_1[k.replace(ignore_prefix, '')] = v
            new_dict = new_dict_1

        did_load = model.load_state_dict(new_dict, strict=False)
        print(did_load, type(model).__name__, load_path)

    return model

class MotionToStaticTeacher(nn.Module):
    """
    Module that takes pretrained motion and/or boundary models,
    runs inference on a pair of images, and uses their outputs to
    compute a motion segment target using MotionSegmentTarget.
    """
    DEFAULT_TARGET_PARAMS = {
        'diffusion_params': None,
        'hp_params': fprop.MotionSegmentTarget.DEFAULT_HP_PARAMS,
        'fp_params': fprop.MotionSegmentTarget.DEFAULT_FP_PARAMS,
        'kp_params': fprop.MotionSegmentTarget.DEFAULT_KP_PARAMS,
        'competition_params': fprop.MotionSegmentTarget.DEFAULT_COMP_PARAMS
    }
    STATIC_STUDENTS = ['eisen', 'thingness', 'centroid']
    MOTION_STUDENTS = ['motion', 'boundary', 'flow']
    def __init__(self,
                 student_model_type='eisen',
                 downsample_factor=2,
                 spatial_resolution=4,
                 motion_resolution=2,
                 motion_beta=10.0,
                 target_from_motion=False,
                 target_motion_thresh=0.5,
                 target_boundary_thresh=0.1,
                 return_intermediates=False,
                 build_flow_target=False,
                 motion_path=None,
                 motion_model_params={},
                 boundary_path=None,
                 boundary_model_params={},
                 flow_path=None,
                 flow_model_params={},
                 target_model_params=DEFAULT_TARGET_PARAMS,
                 parse_paths=True,
                 **kwargs):
        super().__init__()

        self.downsample_factor = downsample_factor
        self.spatial_resolution = spatial_resolution
        self.motion_resolution = motion_resolution
        self.motion_beta = motion_beta
        self.target_from_motion = target_from_motion
        self.target_motion_thresh = target_motion_thresh
        self.target_boundary_thresh = target_boundary_thresh
        self.return_intermediates = return_intermediates
        self.build_flow_target = build_flow_target
        self.parse_paths = parse_paths

        self.motion_model = self._load_motion_model(
            motion_path,
            copy.deepcopy(motion_model_params))
        self.boundary_model = self._load_boundary_model(
            boundary_path,
            copy.deepcopy(boundary_model_params))
        self.flow_model = self._load_flow_model(
            flow_path,
            copy.deepcopy(flow_model_params))
        self.target_model = self._build_target_model(
            copy.deepcopy(target_model_params))
        self.target_model.motion_thresh = self.target_motion_thresh
        self.target_model.boundary_thresh = self.target_boundary_thresh

        self.student_model_type = student_model_type
        self._set_return_type()

    def _load_motion_model(self, path, params):
        return load_model(path,
                          model_class='motion',
                          parse_path=self.parse_paths,
                          **params) if path else None
    def _load_boundary_model(self, path, params):
        return load_model(path,
                          model_class='boundary',
                          parse_path=self.parse_paths,
                          **params) if path else None
    def _load_flow_model(self, path, params):
        return load_model(path,
                          model_class='flow',
                          parse_path=self.parse_paths,
                          **params) if path else None
    def _build_target_model(self, params):
        return fprop.MotionSegmentTarget(
            downsample_factor=self.downsample_factor,
            target_from_motion=self.target_from_motion,
            build_flow_target=self.build_flow_target,
            adj_from_motion=True,
            spatial_resolution=self.spatial_resolution,
            motion_resolution=self.motion_resolution,
            **params)

    def _set_return_type(self):
        if self.return_intermediates:
            self.target_model.target_type = 'motion_static'
        elif self.student_model_type in self.STATIC_STUDENTS:
            self.target_model.target_type = 'static'
        elif self.student_model_type in self.MOTION_STUDENTS:
            self.target_model.target_type = 'motion'
        else:
            self.target_model.target_type = 'motion_static'

    @staticmethod
    def get_motion_preds(net, img1, img2, iters=12, backward=False, nonlinearity=torch.sigmoid):
        if net is None:
            return (None, None)

        if backward:
            ups_mask, motion_preds = net(img2, img1, test_mode=True, iters=iters)
        else:
            ups_mask, motion_preds = net(img1, img2, test_mode=True, iters=iters)

        motion_preds = (nonlinearity or nn.Identity(inplace=True))(motion_preds)
        return (motion_preds, ups_mask)

    @staticmethod
    def get_boundary_preds(net, img1, img2, iters=12, backward=False, nonlinearity=torch.sigmoid):
        if net is None:
            return (None, None, None, None)

        if backward:
            ups_mask, bound_preds = net(img2, img1, test_mode=True, iters=iters)
        else:
            ups_mask, bound_preds = net(img1, img2, test_mode=True, iters=iters)

        C = bound_preds.shape[-3] # number of channels
        motion_preds = None
        if C == 3:
            bound_preds, orientation_preds = bound_preds.split([1,2], -3)
        elif C == 4:
            motion_preds, bound_preds, orientation_preds = bound_preds.split([1,1,2], -3)
            motion_preds = (nonlinearity or nn.Identity())(motion_preds)
        else:
            raise ValueError("%d isn't the correct number of channels for boundary predictions" % C)
        bound_preds = (nonlinearity or nn.Identity())(bound_preds)
        return (bound_preds, orientation_preds, motion_preds, ups_mask)

    @staticmethod
    def get_flow_preds(net, img1, img2, iters=12, backward=False,
                       resolution=None, beta=10.0, thresh=0.5):
        if net is None:
            return (None, None)

        if backward:
            ups_mask, flow_preds = net(img2, img1, test_mode=True, iters=iters)
        else:
            ups_mask, flow_preds = net(img1, img2, test_mode=True, iters=iters)

        if resolution is not None:
            flow_preds = targets.OpticalFlowTarget.delta_hw_to_discrete_flow(
                flow_preds, resolution=resolution, from_xy=False, z_score=False)

        return (flow_preds, ups_mask)

    def _postproc_target(self, target):
        if self.student_model_type == 'motion':
            return (target['motion'] > self.target_motion_thresh).float()
        elif self.student_model_type == 'boundary':
            b,c = target['boundaries'], target['orientations']
            b = (b > self.target_boundary_thresh).float()
            return torch.cat([b,c], 1)
        elif self.student_model_type == 'flow':
            return target['flow']
        else:
            return target

    def forward(self, img1, img2,
                adj=None,
                bootstrap=True,
                run_kp=True,
                *args, **kwargs):

        motion, m_ups_mask = self.get_motion_preds(
            self.motion_model, img1, img2, iters=kwargs.get('motion_iters', 12)
        )
        boundaries, orientations, _motion, b_ups_mask = self.get_boundary_preds(
            self.boundary_model, img1, img2, iters=kwargs.get('boundary_iters', 12)
        )
        flow, f_ups_mask = self.get_flow_preds(
            self.flow_model, img1, img2, iters=kwargs.get('flow_iters', 12))

        video = torch.stack([img1, img2], 1)
        adj = adj.detach() if adj is not None else adj
        target = self.target_model(
            video=(video / 255.),
            motion=motion,
            boundaries=boundaries,
            orientations=orientations,
            flow=flow,
            adj=adj,
            bootstrap=bootstrap,
            run_kp=run_kp
        )
        if self.return_intermediates:
            static_target, motion_targets = target
            return {
                'video': video,
                'motion': motion_targets['motion'],
                'boundaries': motion_targets['boundaries'],
                'orientations': motion_targets['orientations'],
                'flow': motion_targets['flow'],
                'target': static_target
            }
        elif self.student_model_type in self.STATIC_STUDENTS:
            return target
        elif self.student_model_type in self.MOTION_STUDENTS:
            return self._postproc_target(target)
        elif self.student_model_type is None:
            return target
        else:
            raise ValueError("%s is not a valid student model" %\
                             self.student_model_type)


class FuturePredictionTeacher(nn.Module):
    """
    Module that takes pretrained motion and/or boundary models,
    and optionally pretrained spatial affinities, and uses them to
    create a target for either pixelwise future prediction
    (i.e. optical flow / temporal affinities) or for centroid motion.
    """
    DEFAULT_FP_PARAMS = {
        'num_iters': 200,
        'motion_thresh': 0.5,
        'boundary_thresh': 0.1,
        'beta': 10.0,
        'normalize': True
    }
    DEFAULT_TARGET_PARAMS = {
        'warp_radius': 3,
        'warp_dilation': 1,
        'normalize_features': False,
        'patch_radius': 1,
        'error_func': 'sum',
        'distance_func': None,
        'target_type': 'regression',
        'beta': 10.0
    }

    def __init__(self,
                 downsample_factor=1,
                 target_motion_thresh=0.5,
                 target_boundary_thresh=0.1,
                 concat_rgb_features=True,
                 concat_motion_features=False,
                 concat_boundary_features=True,
                 concat_orientation_features=False,
                 concat_fire_features=True,
                 motion_path=None,
                 motion_model_params={},
                 boundary_path=None,
                 boundary_model_params={},
                 parse_paths=False,
                 fp_params=DEFAULT_FP_PARAMS,
                 target_model_params=DEFAULT_TARGET_PARAMS
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.target_boundary_thresh = target_boundary_thresh
        self.target_motion_thresh = target_motion_thresh
        self._concat_r = concat_rgb_features
        self._concat_m = concat_motion_features
        self._concat_b = concat_boundary_features
        self._concat_c = concat_orientation_features
        self._concat_f = concat_fire_features

        self.parse_paths = parse_paths
        self.motion_model = self._load_motion_model(
            motion_path,
            copy.deepcopy(motion_model_params))
        self.boundary_model = self._load_boundary_model(
            boundary_path,
            copy.deepcopy(boundary_model_params))
        self.FP = fprop.FirePropagation(
            downsample_factor=self.downsample_factor,
            compute_kp_args=False,
            **copy.deepcopy(fp_params))
        self.target_model = self._build_target_model(
            copy.deepcopy(target_model_params))

    def _load_motion_model(self, path, params):
        return load_model(path,
                          model_class='motion',
                          parse_path=self.parse_paths,
                          **params) if path else None
    def _load_boundary_model(self, path, params):
        return load_model(path,
                          model_class='boundary',
                          parse_path=self.parse_paths,
                          **params) if path else None
    def _build_target_model(self, params):
        return targets.FuturePredictionTarget(
            **params)

    def _get_features(self, x, y, adj=None, *args, **kwargs):
        motion, m_ups_mask = MotionToStaticTeacher.get_motion_preds(
            self.motion_model, x, y, iters=kwargs.get('motion_iters', 12)
        )
        b_preds = MotionToStaticTeacher.get_boundary_preds(
            self.boundary_model, x, y, iters=kwargs.get('boundary_iters', 12)
        )
        boundaries, orientations, _motion, b_ups_mask = b_preds
        fire = self.FP(
            motion=motion,
            boundaries=boundaries,
            orientations=orientations
        )
        orientations = orientations * (boundaries > self.target_boundary_thresh).float()
        return (self.ds(motion), self.ds(boundaries), self.ds(orientations), fire)

    def forward(self, img1, img2, adj=None, *args, **kwargs):

        self.ds = (lambda x: F.avg_pool2d(
            x, self.downsample_factor, stride=self.downsample_factor)) \
            if self.downsample_factor > 1 else (lambda x: x)
        _to_movie = lambda z: torch.stack([z[0], z[1]], 1)
        rgb = torch.stack([self.ds(img1 / 255.), self.ds(img2 / 255.)], 1)
        motion, boundaries, orientations, fire = map(
            _to_movie,
            zip(self._get_features(img1, img2), self._get_features(img2, img1))
        )
        features = []
        if self._concat_r:
            features.append(rgb)
        if self._concat_m:
            features.append(motion)
        if self._concat_b:
            features.append(boundaries)
        if self._concat_c:
            features.append(orientations)
        if self._concat_f:
            features.append(fire)
        features = torch.cat(features, -3)
        target = self.target_model(features)[0][:,0] # [B,2,H',W']
        interior_mask = torch.logical_or(
            fire[:,0] > 0, boundaries[:,0] > self.target_boundary_thresh).float()
        target = target * (motion[:,0] > self.target_motion_thresh).float() * interior_mask
        boundary_mask = (boundaries[:,0] > self.target_boundary_thresh).float()
        return (target, interior_mask, boundary_mask)

class StaticToMotionTeacher(nn.Module):
    """
    Compute centroid features from a set of tracked segments
    and create an optical flow target from them.
    """
    DEFAULT_CENTROID_PARAMS = {
        'local_radius': 4
    }
    DEFAULT_TARGET_PARAMS = {
        'warp_radius': 3,
        'warp_dilation': 1,
        'normalize_features': False,
        'patch_radius': 0,
        'error_func': None,
        'distance_func': None,
        'target_type': 'regression',
        'beta': 1.0
    }
    def __init__(self,
                 local_centroids=False,
                 local_strides=[1,2,4,8],
                 target_motion_thresh=0.5,
                 filter_motion=False,
                 centroid_model_params=DEFAULT_CENTROID_PARAMS,
                 target_model_params=DEFAULT_TARGET_PARAMS,
                 *args,
                 **kwargs):
        super().__init__()
        if local_centroids:
            self.local_strides = local_strides or []
        else:
            self.local_strides = []
        self.target_motion_thresh = target_motion_thresh
        self.filter_motion = filter_motion
        self.centroid_target = self._build_centroid_target(
            copy.deepcopy(centroid_model_params))

        self.future_target = self._build_future_target(
            copy.deepcopy(target_model_params))

    def _build_centroid_target(self, params):
        return targets.CentroidTarget(
            local_strides=self.local_strides, # if not [], global centroid
            compute_offsets=True,
            normalize=True,
            scale_to_px=True,
            return_masks=True,
            thresh=0.5,
            **params)

    def _build_future_target(self, params):
        return targets.FuturePredictionTarget(**params)

    def _segments_to_masks(self, segments):

        ## do some filtering here

        M = segments.amax().item() + 1
        B,T,H,W = segments.shape
        masks = F.one_hot(segments, num_classes=M)
        masks = masks.view(B*T,H,W,M).permute(0,3,1,2).float()

        self.B, self.T, self.H, self.W = B,T,H,W
        self.BT = self.B*self.T
        return masks

    def _filter_motion_with_masks(self, motion, masks):
        num_px = masks.sum((-2,-1), True).clamp(min=1)
        motion = motion.view(self.BT,1,self.H,self.W)
        is_moving = (masks * motion).sum((-2,-1), True) / num_px
        is_moving = (is_moving > self.target_motion_thresh).float() # [B,M,1,1]
        moving_masks = (masks * is_moving) # [B,M,H,W]
        motion = moving_masks.amax(1, True)
        print("moving masks", moving_masks.shape)
        return motion

    def forward(self, segments, motion):
        assert len(segments.shape) == 4, segments.shape
        assert segments.shape[1] == 2, segments.shape # two frames
        masks = self._segments_to_masks(segments)
        if motion is not None and self.filter_motion:
            motion = self._filter_motion_with_masks(motion, masks)
        if motion is not None:
            motion_mask = motion.view(self.BT,1,1,self.H,self.W)
        else:
            motion_mask = torch.ones((1,1,1,1,1)).to(masks.device)
        target, loss_mask = self.centroid_target(masks * motion_mask[:,0])
        if len(self.centroid_target.local_strides):
            target = target * motion_mask[:,0]
            target = target.view(self.B, self.T, len(self.local_strides)*2, *target.shape[-2:])
        else:
            target = target * loss_mask[:,:,None]
            if motion is not None:
                target = target * motion_mask
            target = target.sum(1).view(self.B,self.T,-1,self.H,self.W)
        target, _ = self.future_target(target)
        if motion is not None:
            target = target * motion[:,0:1]
        return target

class CentroidTeacher(nn.Module):
    """
    From a set of masks and a motion mask, construct a thingness and delta_centroids target
    """
    DEFAULT_CENTROID_PARAMS = {
    }
    def __init__(self,
                 target_motion_thresh=0.5,
                 centroid_model_params=DEFAULT_CENTROID_PARAMS,
                 *args,
                 **kwargs):
        super().__init__()
        self.target_motion_thresh = target_motion_thresh
        self.centroid_target = self._build_centroid_target(
            copy.deepcopy(centroid_model_params))

    def _build_centroid_target(self, params):
        return targets.CentroidTarget(
            local_strides=[],
            compute_offsets=True,
            normalize=True,
            scale_to_px=True,
            return_masks=True,
            thresh=self.target_motion_thresh,
            **params)

    def _segments_to_masks(self, segments):
        M = segments.amax().item() + 1
        B,T,H,W = segments.shape
        masks = F.one_hot(segments, num_classes=M)
        masks = masks.view(B*T,H,W,M).permute(0,3,1,2).float()

        self.B, self.T, self.H, self.W = B,T,H,W
        self.BT = self.B*self.T
        return masks

    def _filter_masks_with_motion(self, masks, motion):
        num_px = masks.sum((-2,-1), True).clamp(min=1)
        is_moving = (masks * motion).sum((-2,-1), True) / num_px
        is_moving = (is_moving > self.target_motion_thresh).float() # [B,M,1,1]
        moving_masks = (masks * is_moving) # [B,M,H,W]
        thingness_target = moving_masks.amax(1, True) # [B,1,H,W]
        return (thingness_target, moving_masks)

    def forward(self, segments, motion):
        assert len(segments.shape) == 4, segments.shape
        assert segments.shape[1] == 1, segments.shape # single frame
        assert len(motion.shape) == 5, motion.shape
        assert motion.shape[-3] == 1, motion.shape
        motion = motion[:,0] # [B,1,H,W]
        masks = self._segments_to_masks(segments) # [B,M,H,W]
        thingness_target, masks = self._filter_masks_with_motion(masks, motion)
        offsets_target, masks = self.centroid_target(masks)
        offsets_target = (offsets_target * masks.unsqueeze(2)).sum(1) * thingness_target
        return (offsets_target, thingness_target)

class GroundTruthGroupingTeacher(nn.Module):

    def __init__(self,
                 downsample_factor=2,
                 radius=7):

        super().__init__()
        self.downsample_factor = self.stride = downsample_factor
        self.radius = self.r = radius
        self.k = 2*self.r + 1
        self.K = self.k**2

    def forward(self, segments, stride=None):

        static = False
        if len(segments.shape) == 4:
            static = True
            segments = segments.unsqueeze(1)
        else:
            assert len(segments.shape) == 5, segments.shape
        assert segments.shape[2] == 1, segments.shape # single channel
        if stride is not None:
            self.stride = stride
        segments = segments[...,::self.stride,::self.stride]
        neighbors = fprop.get_local_neighbors(
            segments[:,:,0], radius=self.r, to_image=True) # [B,T,K,H,W]
        adj = (segments == neighbors).float()
        if static:
            adj = adj[:,0]
        return adj

class BipartiteBootNet(nn.Module):
    """
    """
    DEFAULT_INPUTS = {
        k:'images' for k in ['static', 'centroid', 'dynamic', 'boot']}

    DEFAULT_BOOT_PARAMS = {
        'target_model_params': MotionToStaticTeacher.DEFAULT_TARGET_PARAMS,
        'target_motion_thresh': 0.5,
        'target_boundary_thresh': 0.5
    }
    DEFAULT_GROUPING_PARAMS = {
        'num_iters': 10,
        'radius_temporal': 3,
        'adj_temporal_thresh': None
    }
    DEFAULT_TRACKING_PARAMS = {
        'num_masks': 32,
        'compete_thresh': 0.2,
        'num_competition_rounds': 2,
        'selection_strength': 0
    }
    DEFAULT_STATIC_TO_MOTION_PARAMS = {
    }
    def __init__(self,
                 student_model_type=None,
                 target_model_params=DEFAULT_STATIC_TO_MOTION_PARAMS,
                 input_keys=DEFAULT_INPUTS,
                 dynamic_path=None,
                 dynamic_params={},
                 static_path=None,
                 static_params={},
                 centroid_path=None,
                 centroid_params={},
                 boot_paths={'flow_path': None},
                 parse_paths=True,
                 boot_params=DEFAULT_BOOT_PARAMS,
                 grouping_params=DEFAULT_GROUPING_PARAMS,
                 tracking_params=DEFAULT_TRACKING_PARAMS,
                 input_normalization=None,
                 downsample_factor=2,
                 static_resolution=4,
                 dynamic_resolution=2,
                 random_dimensions=0,
                 grouping_window=2,
                 tracking_step_size=1
    ):
        super().__init__()
        self.parse_paths = parse_paths

        ## preprocessing and downsampling
        self.input_keys = input_keys
        self.downsample_factor = self.stride = downsample_factor
        self.input_normalization = input_normalization or nn.Identity()

        ## dimensions for plateau map
        self._set_plateau_dimensions(
            static_resolution,
            dynamic_resolution,
            random_dimensions
        )

        ## load models
        self._load_models(
            boot_paths=boot_paths, boot_params=boot_params,
            dynamic_path=dynamic_path, dynamic_params=dynamic_params,
            static_path=static_path, static_params=static_params,
            centroid_path=centroid_path, centroid_params=centroid_params
        )

        ## how to slice input video and do spacetime grouping + tracking
        self.Group = self.build_grouping_model(grouping_params)
        self.Track = self.build_tracking_model(tracking_params)
        self.T_group = grouping_window
        self.T_track = tracking_step_size

        ## how to output a training target
        self.student_model_type = student_model_type
        self.build_target_model(target_model_params)

    @property
    def student_model_type(self):
        return self._student_model_type
    @student_model_type.setter
    def student_model_type(self, mode):
        self._student_model_type = mode

    def _set_plateau_dimensions(
            self,
            static_res=None,
            dynamic_res=None,
            random_dims=None
    ):
        self.static_resolution = static_res
        self.dynamic_resolution = dynamic_res
        self.Q_random = random_dims or 0

        self.Q_static = (static_res**2) if static_res else None
        self.Q_dynamic = (dynamic_res**2) if dynamic_res else None
        self.Q = (self.Q_static or 1) * (self.Q_dynamic or 1) + self.Q_random

        print("Plateau map dimensions: [stat %d, dyn %d, rand %d, total %d]" %\
              (self.Q_static or 0, self.Q_dynamic or 0, self.Q_random, self.Q))


    def _load_models(self,
                     boot_paths, boot_params,
                     dynamic_path, dynamic_params,
                     static_path, static_params,
                     centroid_path, centroid_params):
        self._load_boot_model(boot_paths, boot_params)
        self._load_dynamic_model(dynamic_path, dynamic_params)
        self._load_static_model(static_path, static_params)
        self._load_centroid_model(centroid_path, centroid_params)

    def _load_boot_model(self, paths, params):
        self.boot_model = MotionToStaticTeacher(
            student_model_type=None,
            motion_path=paths.get('motion_path', None),
            boundary_path=paths.get('boundary_path', None),
            flow_path=paths.get('flow_path', None),
            downsample_factor=self.downsample_factor,
            spatial_resolution=self.static_resolution,
            motion_resolution=self.dynamic_resolution,
            target_from_motion=False,
            build_flow_target=True,
            parse_paths=self.parse_paths,
            **params
        ) if (paths is not None) else None

    def _load_dynamic_model(self, path, params):

        ## defaults to boot model if there's no path to restore from
        if path is None:
            self.dynamic_model = self.boot_model
            self._dynamic_is_boot_model = True
        else:
            self.dynamic_model = load_model(path,
                                            model_class='flow',
                                            parse_path=self.parse_paths,
                                            **params)
            self._dynamic_is_boot_model = False

    def _load_static_model(self, path, params):
        if path == 'ground_truth':
            print("USING GT AFFINITIES, downsample %d" % self.downsample_factor)
            self.static_model = GroundTruthGroupingTeacher(
                downsample_factor=self.downsample_factor,
                **params)
        elif path is None:
            self.static_model = None
        elif 'eisen' in path:
            self.static_model = load_model(model_class='eisen',
                                           load_path=path,
                                           ignore_prefix='student.',
                                           parse_path=False,
                                           local_affinities_only=True,
                                           **params)

    def _load_centroid_model(self, path, params):
        if path is not None:
            raise NotImplementedError("Need method for loading centroid model")
        self.centroid_model = None

    def build_grouping_model(self, grouping_params):

        return fprop.SpacetimeLocalKProp(**grouping_params)

    def build_tracking_model(self, tracking_params):

        tracker = fprop.ObjectTracker(**tracking_params)
        self.num_competition_rounds = tracker.num_competition_rounds + 0
        return tracker

    def build_target_model(self, target_model_params):
        if self.student_model_type == 'flow_centroids':
            self.target_model = StaticToMotionTeacher(
                local_centroids=False,
                **target_model_params)
        elif self.student_model_type == 'flow':
            self.target_model = StaticToMotionTeacher(
                local_centroids=True,
                **target_model_params)
        elif self.student_model_type == 'centroids':
            self.target_model = CentroidTeacher(
                **target_model_params)
        else:
            self.target_model = None

    @staticmethod
    def get_affinity_preds(net, img1, nonlinearity=None, **kwargs):
        if net is None:
            return None
        else:
            affinities = (nonlinearity or nn.Identity())(net(img1, **kwargs))
            if isinstance(affinities, (list, tuple)):
                affinities = affinities[0]
            return affinities

    @staticmethod
    def get_centroid_preds(net, img1, nonlinearity=None):
        if net is None:
            return None
        else:
            raise NotImplementedError("Need to get centroids")

    @staticmethod
    def get_temporal_preds(net, img1, img2, bootstrap=True, **kwargs):
        if net is None:
            return (None, None)

        if isinstance(net, MotionToStaticTeacher):
            print("using a bootnet to predict flow")
            net.target_model.target_type = 'motion'
            net.return_intermediates = False
            net.student_model_type = 'flow'

        flow_fwd = net(img1, img2, bootstrap=bootstrap, **kwargs)
        flow_bck = net(img2, img1, bootstrap=bootstrap, **kwargs)

        print("flow fwd", flow_fwd.shape)
        print("flow bck", flow_bck.shape)

        return (flow_fwd, flow_bck)

    @staticmethod
    def get_boot_preds(net, img1, img2,
                       bootstrap=True,
                       get_backward_flow=False,
                       **kwargs):
        if net is None:
            return (None, None, None, None)

        assert isinstance(net, MotionToStaticTeacher), type(net).__name__
        net.target_model.target_type = 'motion_static'
        net.return_intermediates = False
        net.student_model_type = None

        h0, adj, activated, flow_fwd, motion_mask = net(img1, img2,
                                                        bootstrap=bootstrap,
                                                        run_kp=False,
                                                        **kwargs)

        flow_bck = None
        if get_backward_flow:
            flow_bck, motion_mask_bck = net(img2, img1,
                                            bootstrap=bootstrap,
                                            run_kp=False,
                                            **kwargs)[-2:]


        return (
            h0, adj, activated,
            flow_fwd, flow_bck,
            motion_mask)

    def _set_shapes(self,
                    video,
                    grouping_window=None,
                    tracking_step_size=None,
                    stride=None
    ):
        if stride is not None:
            self.stride = self.downsample_factor = stride
        if grouping_window is not None:
            self.T_group = grouping_window
        if tracking_step_size is not None:
            self.T_track = tracking_step_size

        self.dsample = lambda x: F.avg_pool2d(
            x, self.stride, stride=self.stride)

        shape = video.shape
        self.B, self.T, _ , self.H_in, self.W_in = shape
        self.is_static = (self.T == 1)
        self.BT = self.B*self.T
        self._T = self.T - 1
        self._BT = self.B*self._T
        self.H, self.W = [self.H_in // self.stride, self.W_in // self.stride]
        self.size = [self.H, self.W]
        self.size_in = [self.H_in, self.W_in]

        assert self.T >= self.T_group, \
            "Input video must be at least length of a grouping window, but" +\
            "T_video = %d and T_group = %d" % (self.T, self.T_group)

        assert self.T_track <= self.T_group, (self.T_track, self.T_group)

    def _normalize(self, video):

        if video.dtype == torch.uint8:
            video = video.float()

        return self.input_normalization(video)

    def _compute_temporal_slices(self):

        self.temporal_slices = [
            [t, t+self.T_group] for t in range(0, self.T - self.T_group + 1, self.T_track)
        ]
        # print("temporal slices", self.temporal_slices)

    def compute_grouping_inputs(
            self,
            video,
            static_params={},
            dynamic_params={},
            boot_params={},
            centroid_params={}
    ):
        """
        Pass each pair of frames into the proper networks to compute
        spatial affinities, temporal affinities, and KP init.

        Then run SpacetimeKP to get a plateau map tensor and Competition
        to get the actual groups and the quantities needed to initialize
        the next window and track objects.
        """
        T = self._get_video(video).shape[1] # could be less than T_group for last window
        _s_inp = lambda t: self._get_frames(video, t, 1, 'static')
        _b_inp = lambda t: self._get_frames(video, t, 2, 'boot')

        if T == 1: # only adj_static
            assert self._static_model is not None
            return (
                self.get_centroid_preds(video[:,0], **centroid_params),
                self.get_affinity_preds(video[:,0], **static_params),
                None,
                None,
                None,
                None
            )

        h0, adj, activated, flow_fwd, flow_bck, motion = [], [], [], [], [], []
        for t in range(T-1): # for each frame pair
            img1, img2 = _b_inp(t)

            ## per-image outputs
            adj_space_t = self.get_affinity_preds(
                self.static_model,
                *[x.detach() for x in _s_inp(t)],
                **static_params)
            h0_space_t = self.get_centroid_preds(
                self.centroid_model, img1, **centroid_params)

            ## per image-pair outputs
            if (self._dynamic_is_boot_model): # use bootnet to get all SKP inputs
                boot_preds = self.get_boot_preds(
                    self.boot_model, img1, img2,
                    get_backward_flow=True,
                    **boot_params)
                h0_t, adj_space_t_boot, activated_t = boot_preds[:3]
                flow_fwd_t, flow_bck_t, motion_fwd_t = boot_preds[3:6]
                if adj_space_t is None:
                    adj_space_t = adj_space_t_boot
            else: # use a separate dynamic model to get the temporal SKP inputs
                flow_fwd_t, flow_bck_t = self.get_temporal_preds(
                    self.dynamic_model, img1, img2, **dynamic_params)
                raise NotImplementedError("build h0 and activated")

            h0.append(h0_t)
            adj.append(adj_space_t)
            activated.append(activated_t)
            flow_fwd.append(self.dsample(flow_fwd_t))
            flow_bck.append(self.dsample(flow_bck_t))
            motion.append(motion_fwd_t)

            ## last frame
            if (t+1) == (T-1):
                adj_space_t = self.get_affinity_preds(
                    self.static_model,
                    *[x.detach() for x in _s_inp(t+1)],
                    **static_params)

                boot_preds = self.get_boot_preds(
                    self.boot_model, img2, img1,
                    get_backward_flow=False,
                    **boot_params)
                h0_t, adj_space_t_boot, activated_t, _, _, motion_bck_t = boot_preds
                if adj_space_t is None:
                    adj_space_t = adj_space_t_boot

                h0.append(h0_t)
                adj.append(adj_space_t)
                activated.append(activated_t)
                motion.append(motion_bck_t)

        adj = torch.stack(adj, 1)
        h0 = torch.stack(h0, 1)
        flow_fwd = torch.stack(flow_fwd, 1)
        flow_bck = torch.stack(flow_bck, 1)
        activated = torch.stack(activated, 1)
        motion = torch.stack(motion, 1)

        return (h0, adj, activated, flow_fwd, flow_bck, motion)

    def compute_initial_segments(self, plateau,
                                 motion_mask=None,
                                 num_rounds=None):

        B,T,H,W,Q = plateau.shape
        if num_rounds is None:
            self.Track.num_competition_rounds = self.num_competition_rounds
        if motion_mask is not None:
            motion = motion_mask[:,:,0,:,:,None] # [B,T,H,W,1]
            plateau = torch.cat([plateau * motion, 1 - motion], -1)

        ## run competition at the midpoint of the group
        t_comp = T // 2
        masks_t, positions_t, alive_t, pointers_t = self.Track(
            plateau[:,t_comp:t_comp+1])[:4]
        segments_t = masks_t.argmax(-1)

        self.tracked_objects = {
            'positions': positions_t,
            'pointers': pointers_t,
            'alive': alive_t
        }

        segments = [segments_t]
        if t_comp > 0:
            self.Track.num_competition_rounds = 0
            masks_pre = self.Track(
                plateau[:,:t_comp],
                agents=positions_t.repeat(1,t_comp,1,1),
                alive=alive_t.repeat(1,t_comp,1,1),
                phenotypes=pointers_t.repeat(1,t_comp,1,1),
                compete=False)[0]
            segments_pre = masks_pre.argmax(-1)
            segments.insert(0, segments_pre)

        if T > (t_comp + 1):
            T_post = T - t_comp - 1
            masks_post = self.Track(
                plateau[:,-T_post:],
                agents=positions_t.repeat(1,T_post,1,1),
                alive=alive_t.repeat(1,T_post,1,1),
                phenotypes=pointers_t.repeat(1,T_post,1,1),
                compete=False)[0]
            segments_post = masks_post.argmax(-1)
            segments.insert(-1, segments_post)

        segments = torch.cat(segments, 1)
        self.Track.num_competition_rounds = self.num_competition_rounds
        return segments

    def compute_overlap_h0(self, plateau, segments, h0_new):
        T_prev, T_new = plateau.shape[1], h0_new.shape[1]
        T_overlap = T_prev - self.T_track
        plateau = plateau[:,-T_overlap:].view(-1, *plateau.shape[2:])
        segments = segments[:,-T_overlap].view(-1, *segments.shape[2:])
        masks = F.one_hot(segments, num_classes=segments.amax().item()+1).float()
        alive = masks.amax((1,2))[...,None]

        plateau_flat, _ =  fprop.ObjectTracker.flatten_plateau_with_masks(
            plateau, masks, alive, flatten_masks=False

        )
        plateau_flat = plateau_flat.view(-1, T_overlap, *plateau_flat.shape[1:])
        h0_overlap = torch.cat([
            plateau_flat.permute(0,1,4,2,3),
            h0_new[:,T_overlap:]
        ], 1)
        return h0_overlap

    def group_tracked_inputs(self,
                             segments_prev,
                             h0,
                             adj,
                             activated,
                             fwd_flow,
                             bck_flow,
                             motion_mask):


        B,T = h0.shape[:2]
        T_overlap = T - self.T_track
        T_new = T - T_overlap

        ## all nodes in the overlapping time slices are already activated
        activated = torch.cat([
            torch.ones_like(activated[:,:T_overlap]),
            activated[:,T_overlap:]], 1)

        plateau_new = self.Group(
            *[x.detach() for x in [
                h0, adj, activated, fwd_flow, bck_flow]],
            motion_mask.detach() if motion_mask is not None else None
        )

        if motion_mask is not None:
            motion = motion_mask[:,:,0,:,:,None]
            _plateau_new = torch.cat([plateau_new * motion, 1 - motion], -1)
        else:
            _plateau_new = plateau_new

        self.Track.num_competition_rounds = 1
        segments_new = self.Track(
            _plateau_new[:,T_overlap:],
            agents=self.tracked_objects['positions'].repeat(1,T_new,1,1),
            alive=self.tracked_objects['alive'].repeat(1,T_new,1,1),
            phenotypes=self.tracked_objects['pointers'].repeat(1,T_new,1,1),
            compete=False,
            update_pointers=True,
            yoke_phenotypes_to_agents=False,
            noise=0)[0].argmax(-1)

        self.Track.num_competition_rounds = self.num_competition_rounds
        segments = torch.cat([
            segments_prev[:,-T_overlap:], segments_new], 1)

        return plateau_new, segments, segments_new

    def _get_video(self, video, img_key='images'):
        if hasattr(video, 'keys'):
            return video[img_key]
        else:
            assert isinstance(video, torch.Tensor)
            return video

    def _get_input(self, video, model_key='boot'):
        if hasattr(video, 'keys'):
            k = self.input_keys.get(model_key, 'images')
            if k != 'images':
                print("WARNING! USING %s AS INPUT, COULD BE GT" % k)
            return video[k]
        else:
            assert isinstance(video, torch.Tensor)
            return video

    def _get_all_inputs(self, video, tstart, tend):
        return {k:video[k][:,tstart:tend] for k in video.keys()}

    def _get_frames(self, video, frame, length=2, model_key='boot'):
        return torch.unbind(
            self._get_input(video, model_key)[:,frame:frame+length], 1)

    def _get_target(self,
                    segments,
                    h0,
                    adj,
                    activated,
                    fwd_flow,
                    bck_flow,
                    motion_mask):
        if self.student_model_type in ['flow', 'flow_centroids']:
            target = self.target_model(segments, motion_mask)
        elif self.student_model_type in ['centroids']:
            centroid_offsets, thingness = self.target_model(segments[:,0:1], motion_mask)
            target = (centroid_offsets, thingness)
        else:
            raise NotImplementedError("%s is not implemented as a training mode for BBNet" % self.student_model_type)
        return target

    @staticmethod
    def segments_to_masks(segments, min_area=None):

        M = segments.amax().item() + 1
        B,T,H,W = segments.shape
        masks = F.one_hot(segments, num_classes=M)
        if min_area is not None:
            num_px = masks.sum((2,3), True)
            area_mask = (num_px > min_area).float()
            masks = masks * area_mask
        return masks

    @staticmethod
    def stitch_segment_movies(segs1, segs2, overlap_window=1, min_area=None):
        B,T1,H,W = segs1.shape
        T2 = segs2.shape[1]
        Toverlap = overlap_window
        masks1 = BipartiteBootNet.segments_to_masks(
            segs1[:,-Toverlap:], min_area=min_area)
        masks2 = BipartiteBootNet.segments_to_masks(
            segs2[:,:Toverlap], min_area=min_area)
        ious = competition.compute_pairwise_overlaps(
            masks1.view(B,Toverlap*H*W,-1),
            masks2.view(B,Toverlap*H*W,-1)
        )
        print(masks1.shape, masks2.shape, ious.shape)

        ## to be a match, iou, has to be at least 0.5
        ## this guarantees that each mask in segs2 has
        ## at most one "parent" in segs1.
        best_ious, parents = ious.max(-2)

        ## figure out new index assignments for segs2
        new_segs = []
        for b in range(B):
            inds = torch.unique(segs2[b,:Toverlap])
            n_segs = segs1[b,-Toverlap].amax() + 1
            p_inds = parents[b]
            b_ious = best_ious[b]
            ## build a hash for the segments
            start = inds.amax() + 1
            new_segs_b = torch.zeros_like(segs2[b])
            for n,i in enumerate(list(inds)):
                b_iou = b_ious[i].item()
                new_segs_b[segs2[b] == i] = p_inds[i] + start
                # else:
                #     new_segs_b[segs2[b] == i] = start + n_segs + n

            new_segs_b -= start
            new_segs.append(new_segs_b)

        new_segs = torch.stack(new_segs, 0)
        stitched = torch.cat([
            segs1,
            new_segs[:,Toverlap:]
        ], 1)

        return stitched

    def forward(self, video,
                stride=None,
                grouping_window=None,
                tracking_step_size=None,
                mask_with_motion=False,
                **kwargs
    ):
        """
        From an RGB video of shape [B,T,3,Hin,Win], computes:

        - Spatial affinities
            - local affinities [B,T,Ks,H,W] where Ks = (2*radius_spatial+1)**2
            - [global affinities] [B,T,HW,HW] from which global affinities can be sampled

        - Temporal Affinities (equivalent to optical flow) [B,T-1,2Kt,H,W],
              where factor of 2 is for backward temporal affinities

        - [Spacetime plateau map initialization] [B,T,Q_static + Q_dynamic + Q_random,H,W]
              where Q_static, Q_dynamic, and Q_random are dimensions that depend on a pixel's
              predicted object centroid, motion direction, and random states, respectively

        - [Spacetime plateau map activations] [B,T,1,H,W]
              a set of points at which to initialize KP

        In a mature BBNet, the spatial affinities are predicted by a per-frame model (EISEN)
        and the temporal affinities are predicted by a video model (RAFT).

        However, during the initial "boot-up" phase of development, the spatial affinities
        (and other KP inputs) are provided by a pretrained BootNet, which operates on
        dynamic scenes and attempts to group based on motion and oriented boundaries.

        The above representations are fed as inputs into SpacetimeKP and a Competition-based
        tracking module, which together produce a visual group tensor of shape [B,T,H,W] <int>
        """
        self._set_shapes(self._get_video(video),
                         grouping_window,
                         tracking_step_size,
                         stride)
        if not hasattr(video, 'keys'):
            video = {'images': video}
        video['images'] = self._normalize(video['images'])

        ## figure out slice indices for video clips
        self._compute_temporal_slices()
        h0_overlap = None
        for window_idx, (ts, te) in enumerate(self.temporal_slices):
            # print(window_idx, (ts, te))
            grp_inputs = self.compute_grouping_inputs(
                self._get_all_inputs(video, ts, te), **kwargs)
            if (self.static_model is not None) and not mask_with_motion:
                motion_mask = None
            else:
                motion_mask = grp_inputs[-1]
            if window_idx == 0:
                plateau = self.Group(*[x.detach() for x in grp_inputs])
                segments = self.compute_initial_segments(plateau, motion_mask)
                full_segments = [segments]
            else: ## use tracking from previous groups
                h0_new = grp_inputs[0]
                h0_overlap = self.compute_overlap_h0(
                    plateau, segments, h0_new)
                plateau, segments, segments_new = self.group_tracked_inputs(
                    segments, h0_overlap, *grp_inputs[1:-1], motion_mask)
                full_segments.append(segments_new)

        full_segments = torch.cat(full_segments, 1)
        if self.target_model is not None:
            target = self._get_target(full_segments, *grp_inputs)
            return target
        return (grp_inputs, plateau, segments, full_segments)

class KpPrior(nn.Module):
    def __init__(self,
                 centroid_model,
                 thingness_model=None,
                 centroid_kwargs={'iters': 12},
                 thingness_kwargs={'iters': 6},
                 thingness_nonlinearity=torch.sigmoid,
                 thingness_thresh=0.1,
                 resolution=8,
                 randomize_background=True,
                 to_nodes=False,
                 normalize_coordinates=True,
                 norm_p=2.0
    ):
        super().__init__()
        self.centroid_model = self._set_model(
            centroid_model, (CentroidRegressor, {}))
        self.normalize_coordinates = normalize_coordinates

        if thingness_model is not None:
            self.thingness_model = self._set_model(
                thingness_model, (ThingsClassifier, {}))
        else:
            self.thingness_model = None

        ## kwargs for model calls
        self._centroid_kwargs = copy.deepcopy(centroid_kwargs)
        self._thingness_kwargs = copy.deepcopy(thingness_kwargs)

        self.thingness_nonlinearity = thingness_nonlinearity or nn.Identity()
        self.thingness_thresh = thingness_thresh

        ## prior params
        self.resolution = resolution
        self.randomize_background = randomize_background
        self.to_nodes = to_nodes
        self.norm_p = norm_p

    def _set_model(self, model, config=None):
        if isinstance(model, nn.Module):
            return model
        assert isinstance(model, str), model
        model_cls, model_params = config
        model_params['args'] = set_args()
        m = nn.DataParallel(model_cls(**model_params))
        m.load_state_dict(torch.load(model), strict=False)
        return m

    def _get_thingness_mask(self, *args, **kwargs):
        if self.thingness_model is None:
            return None

        call_params = copy.deepcopy(kwargs)
        call_params.update(self._thingness_kwargs)
        thingness_mask = self.thingness_model(*args, **call_params)
        if isinstance(thingness_mask, (list, tuple)):
            thingness_mask = thingness_mask[-1]

        thingness_mask = self.thingness_nonlinearity(thingness_mask)
        if self.thingness_thresh is not None:
            thingness_mask = (thingness_mask > self.thingness_thresh).float()

        return thingness_mask

    @staticmethod
    def get_kp_prior(dcentroids, mask=None, resolution=6, randomize_background=True, to_nodes=False, normalize_coordinates=True, norm_p=2.0):
        B,_,H,W = dcentroids.shape
        if normalize_coordinates:
            norm = torch.tensor([(H-1.)/2., (W-1.)/2.], device=dcentroids.device)
            dcentroids = dcentroids / norm.view(1,2,1,1)
        coords = targets.CentroidTarget.coords_grid(
            batch=B, size=[H,W], device=dcentroids.device,
            normalize=True, to_xy=False)

        dcentroids = dcentroids * (mask if mask is not None else 1.)
        kp_prior = targets.CentroidTarget.hw_to_discrete_position(
            dcentroids.view(B, 2, H, W) + coords,
            from_xy=False, resolution=resolution, norm_p=norm_p
        )

        if randomize_background and (mask is not None):
            K = resolution**2
            background = torch.rand((B,K,H,W), device=kp_prior.device).softmax(1)
            kp_prior = mask * kp_prior + (1 - mask) * background

        if to_nodes:
            kp_prior = kp_prior.view(B, -1, H*W).transpose(1, 2)

        return kp_prior

    def forward(self, *args, **kwargs):

        call_params = copy.deepcopy(kwargs)
        call_params.update(self._centroid_kwargs)
        dcentroid_preds = self.centroid_model(*args, **call_params)
        if isinstance(dcentroid_preds, (list, tuple)):
            dcentroid_preds = dcentroid_preds[-1]

        thingness_mask = self._get_thingness_mask(*args, **kwargs)

        kp_prior = self.get_kp_prior(
            dcentroid_preds,
            mask=thingness_mask,
            resolution=self.resolution,
            randomize_background=self.randomize_background,
            to_nodes=self.to_nodes,
            normalize_coordinates=self.normalize_coordinates,
            norm_p=self.norm_p
        )

        return kp_prior

if __name__ == '__main__':

    eisen = load_model(model_class='eisen', load_path='./checkpoints/80000_eisen_teacher_v1_128_bs4.pth', ignore_prefix='student.',
                       stem_pool=False,
                       subsample_affinity=True,
                       local_window_size=25,
                       local_affinities_only=True,
    )
    # print(eisen)
    print(sum([v.numel() for v in eisen.parameters()]))
    eisen.cuda().eval()

    # args = get_args()

    # motion_params = {
    #     'small': False,

    # }
    # boundary_params = {
    #     'small': False,
    #     'static_input': False,
    #     'orientation_type': 'regression'
    # }

    # # target_net = nn.DataParallel(MotionToStaticTeacher(
    # #     downsample_factor=4,
    # #     motion_path=args.motion_path,
    # #     motion_model_params=motion_params,
    # #     boundary_path=args.boundary_path,
    # #     boundary_model_params=boundary_params
    # # ), device_ids=args.gpus)
    # # target_net.eval()

    # bbnet = BipartiteBootNet().cuda()
    video = torch.rand(1,3,256,256) * 255.0
    out, loss, segments = eisen(video.cuda(), to_image=True, local_window_size=15)
    print("out", out.shape)
    # out = bbnet(video)

    # for i in range(100):
    #     img1 = torch.rand((1,3,512,512))
    #     img2 = torch.rand((1,3,512,512))
    #     target = target_net(img1.cuda(), img2.cuda())
    #     print(target.shape, target.dtype, torch.unique(target))
