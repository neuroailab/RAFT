from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raft import RAFT
import dorsalventral.models.bootnet as bootnet
import dorsalventral.models.layers as layers
import dorsalventral.models.targets as targets

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

CentroidMaskTarget = partial(targets.CentroidTarget, normalize=True, return_masks=True)

class KpPrior(nn.Module):
    def __init__(self,
                 centroid_model,
                 thingness_model=None,
                 thingness_nonlinearity=torch.sigmoid,
                 thingness_thresh=0.1,
                 resolution=8,
                 randomize_background=True,
                 to_nodes=False,
                 normalize_coordinates=True
    ):
        super().__init__()
        self.centroid_model = centroid_model
        self.normalize_coordinates = normalize_coordinates

        if thingness_model is not None:
            self.thingness_model = thingness_model
        else:
            self.thingness_model = None

        self.thingness_nonlinearity = thingness_nonlinearity or nn.Identity()
        self.thingness_thresh = thingness_thresh

        ## prior params
        self.resolution = resolution
        self.randomize_background = randomize_background
        self.to_nodes = to_nodes

    def _get_thingness_mask(self, *args, **kwargs):
        if self.thingness_model is None:
            return None

        thingness_mask = self.thingness_model(*args, **kwargs)
        if isinstance(thingness_mask, (list, tuple)):
            thingness_mask = thingness_mask[-1]

        thingness_mask = self.thingness_nonlinearity(thingness_mask)
        if self.thingness_thresh is not None:
            thingness_mask = (thingness_mask > self.thingness_thresh).float()

        return thingness_mask

    @staticmethod
    def get_kp_prior(dcentroids, mask=None, resolution=6, randomize_background=True, to_nodes=False, normalize_coordinates=True):
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
            from_xy=False, resolution=resolution
        )

        if randomize_background and (mask is not None):
            K = resolution**2
            background = torch.rand((B,K,H,W), device=kp_prior.device).softmax(1)
            kp_prior = mask * kp_prior + (1 - mask) * background

        if to_nodes:
            kp_prior = kp_prior.view(B, -1, H*W).transpose(1, 2)

        return kp_prior

    def forward(self, *args, **kwargs):
        dcentroid_preds = self.centroid_model(*args, **kwargs)
        if isinstance(dcentroid_preds, (list, tuple)):
            dcentroid_preds = dcentroid_preds[-1]

        thingness_mask = self._get_thingness_mask(*args, **kwargs)

        kp_prior = self.get_kp_prior(
            dcentroid_preds,
            mask=thingness_mask,
            resolution=self.resolution,
            randomize_background=self.randomize_background,
            to_nodes=self.to_nodes,
            normalize_coordinates=self.normalize_coordinates
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
