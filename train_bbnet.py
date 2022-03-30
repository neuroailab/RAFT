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
from raft import MotionClassifier
from bootraft import (IsMovingTarget,
                      BBNet)
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


# Training/Validation logging
SUM_FREQ = 10
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    mode = getattr(model.module, 'mode', None)
    print("training with mode --- %s" % mode)
    params = model.parameters()
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

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

def get_student_and_teacher(args):
    """Get a teacher and a student model. Teacher defaults to another bbnet"""
    # create and load the student
    cls = MotionClassifier if (args.model.lower() == 'motion') else BBNet
    student = nn.DataParallel(cls(args), device_ids=args.gpus)
    if args.restore_ckpt is not None:
        did_load = student.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print("student", did_load)
    elif args.motion_ckpt is not None:
        state_dict = {k:v for k,v in torch.load(args.motion_ckpt).items()
                      if 'motion_model' in k}
        did_load = student.load_state_dict(state_dict, strict=False)
        print("loaded motion weights only, not", did_load)
    elif args.static_ckpt is not None:
        state_dict = {k:v for k,v in torch.load(args.static_ckpt).items()
                      if 'static_model' in k}
        did_load = student.load_state_dict(state_dict, strict=False)
        print("loaded static weights only, not", did_load)

    student.cuda()

    ## set the training mode of the student
    student.train()
    assert args.train_mode in ['train_static', 'train_motion', 'train_both'], args.train_mode
    student.module.set_mode(args.train_mode)

    # load the teacher
    if args.teacher_ckpt is None:
        return (student, teacher)

    teacher_cls = MotionClassifier if (args.teacher_model.lower() == 'motion') else BBNet
    teacher = nn.DataParallel(teacher_cls(args), device_ids=args.gpus)
    did_load = teacher.load_state_dict(torch.load(args.teacher_ckpt), strict=True)
    print("teacher", did_load)
    teacher.cuda()
    teacher.eval()
    teacher.module.set_mode(args.train_mode)

    return (student, teacher)

def train(args):

    student, teacher = get_student_and_teacher(args)
    if teacher is None:
        assert args.no_aug, "Can't use data augmentation with motion target"
        teacher = nn.DataParallel(IsMovingTarget(
            thresh=None,
            normalize_error=None,
            normalize_features=None,
            get_errors=args.use_motion_loss,
            size=None), device_ids=args.gpus).cuda()

        ## teacher just stacks the images
        def get_target(img1, img2, img0, **kwargs):
            assert img1.dtype == img2.dtype == img0.dtype == torch.float32
            video = torch.stack([img0, img1, img2], 1) / 255.0
            target = (teacher(video) > args.target_thresh).float()
            return (None, target)

    else:
        assert args.no_aug
        def get_target(img1, img2, img0, **kwargs):
            teacher_preds = teacher(img1, img2, iters=args.teacher_iters, test_mode=True)
            target = teacher_preds[-1]
            upsample_mask = None
            if args.upsample:
                upsample_mask = teacher_preds[0]

            return (upsample_mask, target)

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, student)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(student, scheduler)

    VAL_FREQ = args.val_freq
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, image0 = [x.cuda() for x in data_blob[:3]]
            upsample_mask, target = get_target(image1, image2, image0)
            predictions = student(
                image1, image2,
                teacher_motion=target,
                upsample_mask=upsample_mask,
                iters=args.iters
            )

            # get the loss
            if (args.model.lower() == 'motion'):
                loss = student.compute_loss(predictions[-1], target, gamma=args.gamma)
            else:
                loss = predictions[-1]
            loss = loss.mean()
            metrics = {'loss': loss.item()}

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(student.state_dict(), PATH)

                results = {}
                # for val_dataset in args.validation:
                #     if val_dataset == 'chairs':
                #         results.update(evaluate.validate_chairs(model.module))
                #     elif val_dataset == 'sintel':
                #         results.update(evaluate.validate_sintel(model.module))
                #     elif val_dataset == 'kitti':
                #         results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                student.train()
                student.module.set_mode(args.train_mode)

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
    parser.add_argument('--train_split', type=str, default='all')
    parser.add_argument('--flow_gap', type=int, default=1)
    parser.add_argument('--filepattern', type=str, default="*", help="which files to train on tdw")
    parser.add_argument('--test_filepattern', type=str, default="*9", help="which files to val on tdw")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--motion_ckpt', help="restore checkpoint for motion")
    parser.add_argument('--static_ckpt', help="restore checkpoint for static")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--val_freq', type=int, default=5000, help='validation and checkpoint frequency')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
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

    ## model class and training
    parser.add_argument('--model', type=str, default='bbnet', help='Model class')
    parser.add_argument('--train_mode', type=str, default='train_static', help='Which mode to train bbnet')
    parser.add_argument('--teacher_model', type=str, default='motion', help='Model class')
    parser.add_argument('--teacher_ckpt', help='checkpoint for a pretrained RAFT. If None, use GT')
    parser.add_argument('--upsample', action='store_true', help='whether to upsample the motion target')
    parser.add_argument('--teacher_iters', type=int, default=6)
    parser.add_argument('--target_thresh', type=float, default=0.1)
    parser.add_argument('--affinity_radius', type=int, default=5)
    parser.add_argument('--training_frames', help="a JSON file of frames to train from")

    if cmd is None:
        args = parser.parse_args()
        print(args)
    else:
        args = parser.parse_args(cmd)
    return args



def load_model(load_path=None,
               motion_load_path=None,
               static_load_path=None,
               model_class=None,
               small=False,
               cuda=False,
               train=False,
               freeze_bn=False,
               **kwargs):

    path = Path(load_path) if load_path else None

    def _get_model_class(name):
        cls = None
        if 'motion' in name:
            cls = MotionClassifier
        elif 'bbnet' in name:
            cls = BBNet
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
        print(did_load, type(model.module).__name__)

    if motion_load_path is not None:
        state_dict = {k:v for k,v in torch.load(motion_load_path).items()
                      if 'motion_model' in k}
        did_load = model.load_state_dict(state_dict, strict=False)
        print("loaded motion weights only, not", did_load)
    elif static_load_path is not None:
        state_dict = {k:v for k,v in torch.load(static_load_path).items()
                      if 'static_model' in k}
        did_load = model.load_state_dict(state_dict, strict=False)
        print("loaded static weights only, not", did_load)

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
