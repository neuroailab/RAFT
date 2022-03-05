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

from torch.utils.data import DataLoader
from raft import RAFT, ThingsClassifier, CentroidRegressor
from bootraft import BootRaft, CentroidMaskTarget, ForegroundMaskTarget
import evaluate
import datasets

def thingness_sequence_loss(thingness_preds, confident_segments, gamma=0.8):
    """
    Train a pixelwise classifier for thingness.

    args
    thingness_preds: List[Tensor], each of shape [B,1,H,W] in range (-inf, inf)
    confident_segments: Segments of shape [B,H,W] <long> where 0 means not confident
    """

    n_preds = len(thingness_preds)
    loss = 0.0

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss_fn = lambda logits, labels: criterion(logits, labels)

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

def centroid_offset_loss(offset_preds, confident_segments, gamma=0.8):
    """
    Train a pixelwise centroid offset predictor. Loss is masked on pixels that have a confident object.

    args
    offset_preds: List[Tensor], each of shape [B,2,H,W] where the values are the (x,y) offsets in pixels.
    confident_segments: Segments of shape [B,H,W] <long> where 0 means not confident
    """

    n_preds = len(offset_preds)
    loss = 0.0

    criterion = CentroidMaskTarget(thresh=0.5)
    coords = criterion.coords_grid(batch=1, size=confident_segments.shape[-2:],
                                   device=confident_segments.device,
                                   normalize=True, to_xy=False)
    conf_masks = _segments_to_masks(confident_segments) # len B list

    target = []
    for b, masks in enumerate(conf_masks):
        centroids, loss_masks = criterion(masks)
        print("centroids", centroids.shape)
        print("loss masks", loss_masks)

if __name__ == '__main__':

    gt_segments = torch.tensor([
        [0,0,0,1,1],
        [0,0,0,1,1],
        [2,2,3,0,0],
        [2,3,3,3,0],
        [2,0,3,0,4]]).long()
    gt_segments = torch.stack([gt_segments, gt_segments * 2], 0)
    print(gt_segments.shape)
    print(gt_segments)
    


