from functools import partial
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raft import RAFT, CentroidRegressor, ThingsClassifier
import dorsalventral.models.bootnet as bootnet
import dorsalventral.models.layers as layers
import dorsalventral.models.targets as targets
import dorsalventral.models.fire_propagation as fprop

import sys

def get_args(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default="chairs", help="determines which dataset to use for training")
    parser.add_argument('--dataset_names', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--val_freq', type=int, default=5000, help='validation and checkpoint frequency')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--full_playroom', action='store_true')
    parser.add_argument('--static_coords', action='store_true')
    parser.add_argument('--max_frame', type=int, default=5)

    ## model class
    parser.add_argument('--model', type=str, default='RAFT', help='Model class')
    parser.add_argument('--teacher_ckpt', help='checkpoint for a pretrained RAFT. If None, use GT')
    parser.add_argument('--teacher_iters', type=int, default=18)
    parser.add_argument('--scale_centroids', action='store_true')
    parser.add_argument('--training_frames', help="a JSON file of frames to train from")

    if cmd is None:
        args = parser.parse_args()
        print(args)
    else:
        args = parser.parse_args(cmd)
    return args

def set_args(adict={}):
    args = get_args("")
    for k,v in adict.items():
        args.__setattr__(k,v)
    return args

class BootRaft(nn.Module):
    """Wraps bootnet implementatino of RAFT"""
    def __init__(self, args, in_dim=3):
        super(BootRaft, self).__init__()
        self.args = args
        self.encoder = layers.RaftVideoEncoder(in_dim=in_dim)
        self.decoder = layers.RaftVideoDecoder(output_video=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):

        x = torch.stack([image1, image2], 1) # [B,2,3,H,W] <torch.float32>
        x = (x / 255.) # our models expect uint8 in [0,255]
        x = self.encoder(x)

        ## set num iters
        self.decoder.num_iters = iters
        self.decoder.output_predictions = not test_mode
        self.decoder.train(not test_mode)
        x = self.decoder(x)

        if test_mode:
            return (self.decoder.delta_coords, x)
        return x

class BBNet(fprop.BipartiteBootNet):
    """Wraps BipartiteBootNet"""
    def __init__(self, args):
        self.args = args
        super(BBNet, self).__init__(
            affinity_radius=args.affinity_radius,
            motion_params={'hidden_dim': 128, 'num_iters': args.iters},
            static_params={'hidden_dim': 128, 'num_iters': args.iters},
            motion_target_params={},
            static_target_params={},
            motion_sequence_loss=True,
            static_sequence_loss=False,
            sequence_loss_gamma=args.gamma,
            mode=args.train_mode
        )


CentroidMaskTarget = partial(targets.CentroidTarget, normalize=True, return_masks=True)
ForegroundMaskTarget = partial(targets.MotionForegroundTarget, resolution=8, num_masks=32)
IsMovingTarget = partial(targets.IsMovingTarget, warp_radius=3)
DiffusionTarget = partial(fprop.DiffusionTarget,
                          warp_radius=3,
                          boundary_radius=5)


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

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'

    net = BootRaft(None).cuda()


    # net = nn.Sequential(
    #     layers.RaftVideoEncoder(in_dim=3),
    #     layers.RaftVideoDecoder(output_video=False)
    # )

    print(layers.num_parameters(net))
    # print("fnet+cnet", layers.num_parameters(net[0]))
    # print("decoder", layers.num_parameters(net[1]))
    # print(net[1])

    x = torch.rand(2,2,3,512,512).cuda()
    y_list = net(x[:,0], x[:,1], iters=3, test_mode=True)
    print(len(y_list))
    print(y_list[0].shape, y_list[1].shape)
