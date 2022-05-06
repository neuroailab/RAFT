from __future__ import print_function, division
import sys
from typing import List
from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter

sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core.teacher_student import TeacherStudent
import evaluate
import datasets
import math
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

MAX_FLOW = 400
SUM_FREQ = 10
VAL_FREQ = 2000


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
    #                                           pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    scheduler = WarmupPolyLR(optimizer=optimizer, max_iters=args.num_steps)

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
        training_str = "[{:6d}, {:6d}, {:6d}, {:10.7f}] ".format(
            self.total_steps+1, self.epoch, self.step+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, epoch, step, metrics):
        self.epoch = epoch
        self.step = step
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

    kwargs = get_model_args(args)

    for k, v in kwargs.items():
        args.__setattr__(k, v)

    model = nn.DataParallel(TeacherStudent(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print('Restore checkpoint from ', args.restore_ckpt)

    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    train_loader, epoch_size = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    add_noise = True

    should_keep_training = True
    epoch = 0
    while should_keep_training:
        epoch += 1
        for i_batch in range(epoch_size // args.batch_size):
            import time
            t1 = time.time()
            try:
                data_blob = iter(train_loader).next()
            except StopIteration:
                train_loader.dataset.reset_iterator()
                data_blob = iter(train_loader).next()
            except Exception as e:
                print("skipping step %d due to %s" % (total_steps, e))
                total_steps += 1
                continue

            optimizer.zero_grad()

            image1, image2, gt_segment, gt_moving, raft_moving = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            loss, metrics = model(image1, image2, gt_segment=None, iters=args.iters, raft_moving=raft_moving)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(epoch, i_batch + 1, metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

                # results = {}
                # for val_dataset in args.validation:
                #     if val_dataset == 'chairs':
                #         results.update(evaluate.validate_chairs(model.module))
                #     elif val_dataset == 'sintel':
                #         results.update(evaluate.validate_sintel(model.module))
                #     elif val_dataset == 'kitti':
                #         results.update(evaluate.validate_kitti(model.module))
                #
                # logger.write_dict(results)

                model.train()
                # if args.stage != 'chairs':
                #     model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


def eval(args):
    kwargs = get_model_args(args)

    for k, v in kwargs.items():
        args.__setattr__(k, v)

    model = nn.DataParallel(TeacherStudent(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print('Restore checkpoint from ', args.restore_ckpt)

    model.cuda()
    model.eval()

    val_loader, epoch_size = datasets.fetch_dataloader(args)

    total_steps = 0

    add_noise = True


    avg_metric = {}

    start = time.time()
    for i_batch, data_blob in enumerate(iter(val_loader)):

        image1, image2, gt_segment, gt_moving, raft_moving = [x.cuda() for x in data_blob]

        loss, metrics = model(image1, image2, gt_segment, iters=args.iters, raft_moving=raft_moving, get_segments=True)

        for k, v in metrics.items():
            if k in avg_metric.keys():
                avg_metric[k].append(v)
            else:
                avg_metric[k] = [v]

        if (i_batch + 1) % 5 == 0:
            for k, v in avg_metric.items():
                print(k, np.nanmean(v), len(v), (time.time() - start) / i_batch)

        #
        # if i_batch > 50:
        #     break

    for k, v in avg_metric.items():
        print(k, np.nanmean(v))


def get_model_args(args):
    # [Params 1]
    # motion_params = {'small': False}
    # boundary_params = {
    #     'small': False,
    #     'static_input': False,
    #     'orientation_type': 'regression'
    # }
    # motion_path = './checkpoints/15000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall-ctd.pth'
    # boundary_path = './checkpoints/40000_boundaryStaticReg-rnd0-tdw-bs4-large-dtarg-nthr0-cthr075-pr1-tds2-fullplayall.pth'

    # teacher_params = {
    #     'downsample_factor': 4,
    #     'motion_path': motion_path,
    #     'motion_model_params': motion_params,
    #     'boundary_path': boundary_path,
    #     'boundary_model_params': boundary_params,
    #     'target_from_motion': True
    # }

    # [Params 2]
    m_path = './checkpoints/72500_motion-rnd1-movi_d-bs2-large-mt05-bt05-flit24-gs1-pretrained-0.pth'
    b_path = './checkpoints/72500_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth'
    f_path = './checkpoints/62500_flowBoundary-rnd1-movi_d-bs2-large-mt05-bt01-flit24-gs1-pretrained-0.pth'

    motion_model_params = {
        'small': 'small' in m_path,
        'gate_stride': 2 if ('gs1' not in m_path) else 1
    }
    boundary_model_params = {
        'small': 'small' in b_path,
        'static_input': 'Static' in b_path,
        'orientation_type': 'regression',
        'gate_stride': 2 if ('gs1' not in b_path) else 1
    }

    teacher_params = {
        'student_model_type': None,
        'downsample_factor': 2,
        'spatial_resolution': 4,
        'motion_resolution': 2,
        'target_from_motion': False,
        'return_intermediates': True,
        'build_flow_target': True,
        'motion_path': m_path,
        'boundary_path': b_path,
        'flow_path': f_path,
        'motion_model_params': motion_model_params,
        'boundary_model_params': boundary_model_params
    }

    student_params = {}
    if 'tdw' in args.stage:
        student_params['affinity_res'] = [128, 128]
    else:
        student_params['affinity_res'] = [64, 64]

    student_params['stem_pool'] = args.stem_pool

    if not args.stem_pool:
        student_params['affinity_res'] = [128, 128]

    kwargs = {
        'teacher_class': args.teacher_class,
        'teacher_params': teacher_params,
        'teacher_load_path': None,
        'student_class': 'eisen',
        'student_params': student_params,
        'student_load_path': None,
    }
    return kwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default="chairs", help="determines which dataset to use for training")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dataset_names', type=str, nargs='+')
    parser.add_argument('--train_split', type=str, default='all')
    parser.add_argument('--flow_gap', type=int, default=1)
    parser.add_argument('--filepattern', type=str, default="*", help="which files to train on tdw")
    parser.add_argument('--test_filepattern', type=str, default="*9", help="which files to val on tdw")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--val_freq', type=int, default=5000, help='validation and checkpoint frequency')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.0000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--full_playroom', action='store_true')
    parser.add_argument('--static_coords', action='store_true')
    parser.add_argument('--max_frame', type=int, default=5)
    parser.add_argument('--model', type=str, default='teacher_student', help='Model class')
    parser.add_argument('--teacher_class', type=str, default='motion_to_static', help='Teacher class')
    parser.add_argument('--training_frames', help="a JSON file of frames to train from")
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--stem_pool', type=int, default=1, help="whether to pool in the backbone stem")

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.eval_only:
        eval(args)
    else:
        train(args)
