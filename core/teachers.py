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

import dorsalventral.models.layers as layers
import dorsalventral.models.targets as targets
import dorsalventral.models.fire_propagation as fprop

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
    model = cls(args)
    if load_path is not None:
        weight_dict = torch.load(load_path)
        new_dict = dict()
        for k in weight_dict.keys():
            if 'module' in k:
                new_dict[k.split('module.')[-1]] = weight_dict[k]
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
        'fp_params': fprop.MotionSegmentTarget.DEFAULT_FP_PARAMS,
        'kp_params': fprop.MotionSegmentTarget.DEFAULT_KP_PARAMS,
        'competition_params': fprop.MotionSegmentTarget.DEFAULT_COMP_PARAMS
    }
    def __init__(self,
                 downsample_factor=4,
                 motion_resolution=4,
                 motion_beta=10.0,
                 target_from_motion=False,
                 return_intermediates=False,
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
        self.motion_resolution, self.motion_beta = motion_resolution, motion_beta
        self.target_from_motion = target_from_motion
        self.return_intermediates = return_intermediates
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
            adj_from_motion=True,
            **params)

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
        else:
            flow_mask = (flow_preds.square().sum(1, True).sqrt() > thresh).float()
            flow_preds = fprop.spatial_moments_to_circular_target(flow_preds, beta)
            static = torch.cat([
                torch.zeros_like(flow_preds[:,:4]),
                torch.ones_like(flow_preds[:,4:5]),
                torch.zeros_like(flow_preds[:,5:])], 1)
            flow_preds = flow_preds * flow_mask + static * (1 - flow_mask)

        return (flow_preds, ups_mask)

    def forward(self, img1, img2, adj=None, *args, **kwargs):

        motion, m_ups_mask = self.get_motion_preds(
            self.motion_model, img1, img2, iters=kwargs.get('motion_iters', 12)
        )
        boundaries, orientations, _motion, b_ups_mask = self.get_boundary_preds(
            self.boundary_model, img1, img2, iters=kwargs.get('boundary_iters', 12)
        )
        flow, f_ups_mask = self.get_flow_preds(
            self.flow_model, img1, img2, iters=kwargs.get('flow_iters', 12),
            resolution=self.motion_resolution, beta=self.motion_beta)
        target = self.target_model(
            video=torch.stack([img1, img2], 1) * 255.0,
            motion=motion,
            boundaries=boundaries,
            orientations=orientations,
            adj=adj
        )
        if not self.return_intermediates:
            return target
        return {
            'motion': motion,
            'boundaries': boundaries,
            'orientations': orientations,
            'flow': flow,
            'motion_upsample_mask': m_ups_mask,
            'boundary_upsample_mask': b_ups_mask,
            'flow_upsample_mask': f_ups_mask,
            'target': target
        }

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
        interior = torch.logical_or(
            fire[:,0] > 0, boundaries[:,0] > self.target_boundary_thresh).float()
        target = target * (motion[:,0] > self.target_motion_thresh).float() * interior
        return target

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
    args = get_args()

    motion_params = {
        'small': False,

    }
    boundary_params = {
        'small': False,
        'static_input': False,
        'orientation_type': 'regression'
    }

    target_net = nn.DataParallel(MotionToStaticTeacher(
        downsample_factor=4,
        motion_path=args.motion_path,
        motion_model_params=motion_params,
        boundary_path=args.boundary_path,
        boundary_model_params=boundary_params
    ), device_ids=args.gpus)
    target_net.eval()

    for i in range(100):
        img1 = torch.rand((1,3,512,512))
        img2 = torch.rand((1,3,512,512))
        target = target_net(img1.cuda(), img2.cuda())
        print(target.shape, target.dtype, torch.unique(target))
