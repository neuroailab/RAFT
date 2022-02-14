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
from bootraft import BootRaft, CentroidMaskTarget
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 10
VAL_FREQ = 5000

# datasets without supervision
SELFSUP_DATASETS = ['robonet', 'dsr']


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, min_flow=0.5, pos_weight=1.0):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid is None:
        valid = (mag < max_flow)
    else:
        valid = (valid >= 0.5) & (mag < max_flow)

    if flow_preds[-1].shape[-3] == 1:
        flow_gt = (mag[:,None] > min_flow).float()
        pos_weight = torch.tensor([pos_weight], device=flow_gt.device)
        loss_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        loss_fn = lambda logits, labels: loss_cls(logits, labels)
    else:
        loss_fn = lambda logits, labels: (logits - labels).abs()
        assert flow_preds[-1].shape[-3] == 2, flow_preds[-1].shape

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        # i_loss = (flow_preds[i] - flow_gt).abs()
        i_loss = loss_fn(flow_preds[i], flow_gt)
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    # epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'loss': flow_loss,
        # 'epe': epe.mean().item(),
        # '1px': (epe < 1).float().mean().item(),
        # '3px': (epe < 3).float().mean().item(),
        # '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def centroid_loss(dcent_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, min_flow=0.5, scale_to_pixels=False):

    n_predictions = len(dcent_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid is None:
        valid = (mag < max_flow)
    else:
        valid = (valid >= 0.5) & (mag < max_flow)
    CT = CentroidMaskTarget(thresh=min_flow)
    centroid, mask = CT(mag[:,None].float()) # centroid
    coords = CT.coords_grid(batch=1, size=mask.shape[-2:], device=centroid.device,
                            normalize=True, to_xy=False)
    target = centroid[...,None] * mask - coords
    if scale_to_pixels:
        H,W = target.shape[-2:]
        target = target * torch.tensor([(H-1.)/2.,(W-1.)/2.], device=target.device).float().view(1,2,1,1)
    num_px = mask.detach().sum(dim=(-2,-1)).clamp(min=1.0)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (dcent_preds[i] - target).square()
        i_loss = (i_loss * mask).sum(dim=(-2,-1)) / num_px
        flow_loss += i_weight * i_loss.mean()

    metrics = {'loss': flow_loss}
    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    if args.model.lower() == 'bootraft':
        model_cls = BootRaft
        print("used BootRaft")
    elif args.model.lower() == 'thingness':
        model_cls = ThingsClassifier
        print("used ThingnessClassifier")
    elif args.model.lower() == 'centroid':
        model_cls = CentroidRegressor
        print("used CentroidRegressor")
    else:
        model_cls = RAFT
        print("used RAFT")
    model = nn.DataParallel(model_cls(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        did_load = model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print(did_load)

    model.cuda()
    model.train()

    if args.stage not in ['chairs', 'tdw', 'robonet']:
        model.module.freeze_bn()

    ## load a teacher model
    selfsup = False

    ## if training on a dataset without gt flow, we __must__ self-supervise
    if args.stage in SELFSUP_DATASETS:
        selfsup = True
        assert args.teacher_ckpt is not None, "You're training on %s that has no gt flow! Pass a teacher checkpoint" % args.stage

    if args.teacher_ckpt is not None:
        selfsup = True
        _small = args.small
        args.small = False
        teacher = nn.DataParallel(RAFT(args), device_ids=args.gpus)
        did_load = teacher.load_state_dict(torch.load(args.teacher_ckpt), strict=False)
        args.small = _small
        print("TEACHER")
        print(did_load)
        teacher.cuda()
        teacher.eval()
        teacher.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = args.val_freq
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            if selfsup:
                image1, image2 = [x.cuda() for x in data_blob[:2]]
                valid = None
            else:
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            if selfsup:
                _, flow = teacher(image1, image2, iters=args.teacher_iters, test_mode=True)

            if args.model.lower() == 'centroid':
                loss, metrics = centroid_loss(flow_predictions, flow, valid, args.gamma, scale_to_pixels=args.scale_centroids)
            else:
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma, pos_weight=args.pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)

                model.train()
                if args.stage not in ['chairs', 'tdw', 'robonet']:
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH

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
    parser.add_argument('--pos_weight', type=float, default=1.0, help='weight for positive bce samples')
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

def load_model(load_path,
               model_class=None,
               small=False,
               cuda=False,
               train=False,
               freeze_bn=False,
               **kwargs):

    path = Path(load_path)

    def _get_model_class(name):
        cls = None
        if 'bootraft' in name:
            cls = BootRaft
        elif 'raft' in name:
            cls = RAFT
        elif 'thing' in name:
            cls = ThingsClassifier
        elif 'centroid' in name:
            cls = CentroidRegressor
        else:
            raise ValueError("Couldn't identify a model class associated with %s" % name)
        return cls

    if model_class is None:
        cls = _get_model_class(path.name)
    else:
        cls = _get_model_class(model_class)
    assert cls is not None, "Wasn't able to infer model class"

    ## get the args
    args = get_args("")
    if small:
        args.small = True
    for k,v in kwargs.items():
        args.__setattr__(k,v)

    # build model
    model = nn.DataParallel(cls(args), device_ids=args.gpus)
    if load_path is not None:
        did_load = model.load_state_dict(torch.load(load_path), strict=False)
        print(did_load)
    if cuda:
        model.cuda()
    model.train(train)
    if freeze_bn:
        model.module.freeze_bn()

    return model

if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
