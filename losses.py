from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Resize

from torch.utils.data import DataLoader
from raft import RAFT, ThingsClassifier, CentroidRegressor
from bootraft import BootRaft, CentroidMaskTarget, ForegroundMaskTarget
import evaluate
import datasets

def thingness_loss(thingness_preds, confident_segments, gamma=0.8):
    """
    Train a pixelwise classifier for thingness.

    args
    thingness_preds: List[Tensor], each of shape [B,1,H,W] in range (-inf, inf)
    confident_segments: Segments of shape [B,H,W] <long> where 0 means not confident
    """

    n_preds = len(thingness_preds)
    loss = 0.0

    size = confident_segments.shape[-2:]
    resize = Resize(size)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss_fn = lambda logits, labels: criterion(resize(logits), labels)

    ## target is any pixel that has a nonzero label
    target = (confident_segments > 0)[:,None].float() # [B,1,H,W] float

    for i in range(n_preds):
        i_weight = gamma ** (n_preds - i - 1)
        i_loss = loss_fn(thingness_preds[i], target)
        loss += (i_weight * i_loss).mean()

    metrics = {'loss': loss.item()}

    return (loss, metrics)

def _segments_to_masks(segs, max_labels=128, background_id=0):
    """[B,H,W] <long> segments to List B list of [K-1,H,W] <float> masks where K = (num unique mask ids)"""

    out = []
    for b, seg in enumerate(list(segs)):
        labels = torch.unique(seg)[:max_labels]
        masks = (labels[:,None,None] == seg)
        masks = torch.cat([masks[0:background_id],
                           masks[background_id+1:]], 0)
        out.append(masks.float())
    return out

def multicentroid_offset_loss(offset_preds, confident_segments, gamma=0.8):
    """
    Train a pixelwise centroid offset predictor. Loss is masked on pixels that have a confident object.

    args
    offset_preds: List[Tensor], each of shape [B,2,H,W] where the values are the (x,y) offsets in pixels.
    confident_segments: Segments of shape [B,H,W] <long> where 0 means not confident
    """

    n_preds = len(offset_preds)
    size = confident_segments.shape[-2:]
    H,W = size
    scale_factor = torch.tensor([(H-1.)/2., (W-1.)/2.])
    scale_factor = scale_factor.view(1,2,1,1,1).float().to(confident_segments.device)

    _resize = Resize(size)
    def resize(logits):
        h,w = logits.shape[-2:]
        scale = torch.tensor([H/h,W/w]).to(logits.device).float()
        return _resize(logits) * scale.view(1,2,1,1)

    loss = 0.0

    criterion = CentroidMaskTarget(thresh=0.5)
    coords = criterion.coords_grid(batch=1, size=size,
                                   device=confident_segments.device,
                                   normalize=True, to_xy=False)
    coords = coords.unsqueeze(2) # [1,2,1,H,w]
    conf_masks = _segments_to_masks(confident_segments) # len B list

    for b, masks in enumerate(conf_masks):
        ## centroids are [1,2,K]
        ## loss masks are [1,K,H,W]
        centroids, loss_masks = criterion(masks[None])
        loss_masks = loss_masks[:,None] # [1,1,K,H,W]
        offset_target = centroids[...,None,None] * loss_masks - coords
        offset_target = offset_target * scale_factor # scale to px
        num_px = loss_masks.detach().sum(dim=(-2,-1)).clamp(min=1.)

        for i in range(n_preds):
            i_weight = gamma ** (n_preds - i - 1)
            preds = resize(offset_preds[i][b:b+1])[:,:,None] # [1,2,1,H,W]
            i_loss = (preds - offset_target).square()
            i_loss = (i_loss * loss_masks).sum(dim=(-2,-1)) / num_px
            loss += i_weight * i_loss.mean()

    metrics = {'loss': loss.item()}

    return (loss, metrics)

def motion_loss(motion_preds, errors, valid, gamma=0.8, loss_scale=1.0):

    n_predictions = len(motion_preds)
    loss = 0.0

    errors_s, errors_m = errors.split([1,1], -3)
    def loss_fn(preds):
        p_motion = torch.sigmoid(preds)
        return (p_motion*errors_m + (1-p_motion)*errors_s).mean()

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = loss_fn(motion_preds[i])
        loss += i_weight * i_loss * loss_scale

    metrics = {'loss': loss.item()}

    return loss, metrics

if __name__ == '__main__':

    gt_segments = torch.tensor([
        [0,0,0,1,1],
        [0,0,0,1,1],
        [2,2,3,0,0],
        [2,3,3,3,0],
        [2,0,3,0,4]]).long()
    gt_segments = torch.stack([gt_segments, gt_segments * 2], 0)
    gt_segments[1,-1,-1] = 0
    print(gt_segments.shape)
    print(gt_segments)

    preds = [2.5 * torch.randn([2,2,5,5]) for _ in range(3)]

    l, m = multicentroid_offset_loss(preds, gt_segments)
    print(m)

    preds = [10 * torch.randn([2,1,20,20]) for _ in range(3)]
    l,m = thingness_loss(preds, gt_segments)
    print(m)
