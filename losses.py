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


    
