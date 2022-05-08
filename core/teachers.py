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

class BipartiteBootNet(nn.Module):
    """
    """
    DEFAULT_BOOT_PARAMS = {
        'target_model_params': MotionToStaticTeacher.DEFAULT_TARGET_PARAMS,
        'target_motion_thresh': 0.5,
        'target_boundary_thresh': 0.5
    }
    DEFAULT_GROUPING_PARAMS = {
        'num_iters': 40,
        'radius_temporal': 3,
        'adj_temporal_thresh': None
    }
    DEFAULT_TRACKING_PARAMS = {
        'num_masks': 32,
        'compete_thresh': 0.2,
        'num_competition_rounds': 2
    }
    def __init__(self,
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

    def _set_plateau_dimensions(
            self,
            static_res=None,
            dynamic_res=None,
            random_dims=None
    ):
        self.static_resolution = static_res or 0
        self.dynamic_resolution = dynamic_res or 0
        self.Q_random = random_dims or 0

        self.Q_static = static_res**2
        self.Q_dynamic = dynamic_res**2
        self.Q = self.Q_static + self.Q_dynamic + self.Q_random

        print("Plateau map dimensions: [stat %d, dyn %d, rand %d, total %d]" %\
              (self.Q_static, self.Q_dynamic, self.Q_random,
               self.Q_static * self.Q_dynamic + self.Q_random))

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
        if path is not None:
            raise NotImplementedError("Need method for loading EISEN")
        self.static_model = None

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

    @staticmethod
    def get_affinity_preds(net, img1, nonlinearity=None):
        if net is None:
            return None
        else:
            raise NotImplementedError("Need to get affinities")

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
            [t, t+self.T_group] for t in range(0, self.T - 1, self.T_track)
        ]
        print("temporal slices", self.temporal_slices)

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
        T = video.shape[1] # could be less than T_group for last window
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
            img1, img2 = video[:,t], video[:,t+1]

            ## per-image outputs
            adj_space_t = self.get_affinity_preds(
                self.static_model, img1, **static_params)
            h0_space_t = self.get_centroid_preds(
                self.centroid_model, img1, **centroid_params)

            ## per image-pair outputs
            if (adj_space_t is None): # use bootnet to get all SKP inputs
                boot_preds = self.get_boot_preds(
                    self.boot_model, img1, img2,
                    get_backward_flow=True,
                    **boot_params)
                h0_t, adj_space_t, activated_t = boot_preds[:3]
                flow_fwd_t, flow_bck_t, motion_fwd_t = boot_preds[3:6]
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
                if self.static_model is not None:
                    adj.append(self.get_affinity_preds(
                        self.static_model, img2, **static_params))
                    raise NotImplementedError("Build h0 and activated for last frame")
                else: # use backward predictions from boot model
                    boot_preds = self.get_boot_preds(
                        self.boot_model, img2, img1,
                        get_backward_flow=False,
                        **boot_params)
                    h0_t, adj_space_t, activated_t, _, _, motion_bck_t = boot_preds
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
        print("t comp", t_comp)
        masks_t, positions_t, alive_t, pointers_t = self.Track(
            plateau[:,t_comp:t_comp+1])[:4]
        segments_t = masks_t.argmax(-1)

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


    def forward(self, video,
                stride=None,
                grouping_window=None,
                tracking_step_size=None):
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
        self._set_shapes(video, grouping_window, tracking_step_size, stride)
        video = self._normalize(video)

        ## figure out slice indices for video clips
        self._compute_temporal_slices()
        h0_overlap = None
        for window_idx, (ts, te) in enumerate(self.temporal_slices):
            print(window_idx, (ts, te))
            grp_inputs = self.compute_grouping_inputs(
                video[:,ts:te])
            motion_mask = grp_inputs[-1]
            for i,v in enumerate(grp_inputs):
                print(i, v.shape)
            if window_idx == 0:
                plateau = self.Group(*[x.detach() for x in grp_inputs])
                print("plateau", plateau.shape)
                segments = self.compute_initial_segments(plateau, motion_mask)
                print("segments", segments.shape, segments.dtype)
            else: ## use tracking from previous groups
                h0_new = grp_inputs[0]
                h0_overlap = self.compute_overlap_h0(
                    plateau, segments, h0_new)
                print("h0 overlap", h0_overlap.shape)
                print("h0", grp_inputs[0].shape)

        return (grp_inputs, plateau, segments, h0_overlap)

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

    # target_net = nn.DataParallel(MotionToStaticTeacher(
    #     downsample_factor=4,
    #     motion_path=args.motion_path,
    #     motion_model_params=motion_params,
    #     boundary_path=args.boundary_path,
    #     boundary_model_params=boundary_params
    # ), device_ids=args.gpus)
    # target_net.eval()

    bbnet = BipartiteBootNet().cuda()
    video = torch.rand(1,8,3,256,256) * 255.0
    out = bbnet(video)

    # for i in range(100):
    #     img1 = torch.rand((1,3,512,512))
    #     img2 = torch.rand((1,3,512,512))
    #     target = target_net(img1.cuda(), img2.cuda())
    #     print(target.shape, target.dtype, torch.unique(target))
