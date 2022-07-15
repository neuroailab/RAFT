from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import copy
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
from torchvision import transforms

from torch.utils.data import DataLoader
from raft import (RAFT,
                  ThingsClassifier,
                  CentroidRegressor,
                  MotionClassifier,
                  BoundaryClassifier,
                  SpatialAffinityDecoder)
from eisen import EISEN

import teachers
import core.utils.utils as utils
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

import dorsalventral.models.bbnet.innate as innate
import dorsalventral.models.bbnet.pathways as pathways
import dorsalventral.models.bbnet.messages as messages
import dorsalventral.models.layers as layers

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
        self.total_steps = args.restore_step
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

    teacher_config = args.teacher_config or {}
    if args.bootstrap:
        student_config = teacher_config
    else:
        student_config = args.student_config or {}

    # create an affinity (coupling) model
    _model = nn.DataParallel(
        teachers.load_model(
            load_path=args.restore_ckpt,
            model_class='affinity',
            affinity_radius=args.affinity_radius,
            small=args.small,
            gate_stride=args.gate_stride,
            static_input=args.static_input,
            affinity_nonlinearity='none',
            parse_path=False),
        device_ids=args.gpus
    )

    # wrap the model
    if args.static_input:
        wrapper = pathways.RaftPerFrameStaticWrapper
    else:
        wrapper = pathways.RaftPerFrameMotionWrapper
    model = wrapper(
        net=_model,
        model_kwargs={'test_mode': not (args.model == 'affinity'), 'iters': args.iters},
        nonlinearity='none',
        postproc_func=None)
    
    model.cuda()
    model.train()
    if args.model != 'affinity':
        model.requires_grad_(False)

    print("Parameter Count: %d" % count_parameters(model))
    print("static input? %s" % args.static_input)

    if args.gate_ckpt is not None:
        _gate_model = nn.DataParallel(
            teachers.load_model(
                load_path=args.gate_ckpt,
                model_class=args.gate_model,
                small=args.small,
                gate_stride=args.gate_stride,
                out_channels=args.out_channels,
                parse_path=False),
            device_ids=args.gpus
        )
        gate_model = pathways.ConvnetPerFrameWrapper(
            net=_gate_model,
            postproc_func=pathways.raft_postproc,
            nonlinearity='sigmoid')
        
        gate_model.requires_grad_(False)
        gate_model.train()
        gate_model.cuda()
        print("Parameter count gate model: %d" % count_parameters(gate_model))
    else:
        gate_model = None
        print("No gate")

    _target_model = nn.DataParallel(
        teachers.load_model(
            load_path=args.teacher_ckpt,
            model_class='motion',
            small=args.small_teacher,
            gate_stride=args.gate_stride),
        device_ids=args.gpus)
    target_model = pathways.RaftVideoWrapper(
        net=_target_model,
        model_kwargs={'test_mode': True, 'iters': args.teacher_iters},
        nonlinearity='sigmoid',
        postproc_func=pathways.raft_postproc,
        stop_grad=True)

    target_model.requires_grad_(False)
    target_model.train()
    target_model.cuda()
    print("Parameter count target model: %d" % count_parameters(target_model))

    if args.model == 'motion':
        _student_model = nn.DataParallel(
            teachers.load_model(
                load_path=args.teacher_ckpt,
                model_class='motion',
                small=args.small_teacher,
                gate_stride=args.gate_stride),
            device_ids=args.gpus)
        student_model = pathways.RaftPerFrameMotionWrapper(
            net=_student_model,
            model_kwargs={'test_mode': False, 'iters': args.iters},
            nonlinearity='none',
            postproc_func=None,
            stop_grad=False)
        student_model.cuda()
        student_model.train()
        student_model.requires_grad_(True)
        print("Parameter count student model: %d" % count_parameters(student_model))        
    else:
        student_model = None


    PBS = nn.DataParallel(
        pathways.PathwayBinarySampler(
            num_samples=args.num_samples,
            thresh=args.pathway_thresh),
        device_ids=args.gpus).cuda()
    
    if args.affinity_nonlinearity == 'softmax_max':
        loss_func = messages.utils.MaskedKLDivLoss(dim=-1)
    elif args.affinity_nonlinearity == 'sigmoid':
        loss_func = messages.utils.MaskedBCELoss(with_logits=True)
    MSG = nn.DataParallel(
        messages.LocalSpatialMessages(
            radius=args.affinity_radius,
            affinity_nonlinearity=args.affinity_nonlinearity,
            loss_func=loss_func,
            num_iters=(1 if student_model is None else args.num_propagation_iters),
            store_every=None,
            mask_background=True,
            num_samples=args.num_samples,
            confidence_thresh=args.target_thresh,
            integration_window=args.integration_window
        ),
        device_ids=args.gpus)

    MSG.cuda()
    MSG.train()

    ## data and optimizer
    train_loader, epoch_size = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(
        args, model=(model if args.model == 'affinity' else student_model))

    total_steps = args.restore_step
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = args.val_freq
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
            video = data_blob['images'].cuda()
            frame = (video.shape[1] - 1) // 2

            affinities_list = model(video)
            motion_preds = target_model(video)
            target, loss_mask = PBS(motion_preds)

            if gate_model is not None:
                gate_preds = gate_model(video)
                gate, _ = PBS(gate_preds)
                loss_mask = loss_mask * gate

            loss = 0.0
            if student_model is None:
                for i, affs in enumerate(affinities_list):
                    _ = MSG(
                        source_latent=target,
                        affinities=affs,
                        mask=loss_mask)
                    i_loss = MSG.module.loss
                    i_weight = args.gamma**(len(affinities_list) - i - 1)
                    loss += i_loss * i_weight
            else:
                loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                student_preds = student_model(video)
                motion_target = MSG(
                    source_latent=target,
                    affinities=affinities_list[-1],
                    mask=loss_mask)
                motion_target = (motion_target > 0.5).float()
                for i, pred in enumerate(student_preds):
                    i_loss = loss_fn(pred, motion_target).mean()
                    i_weight = args.gamma**(len(student_preds) - i - 1)
                    loss += i_loss * i_weight

                if total_steps == 1:
                    SV_PATH = 'checkpoints/motion_targets.pt'
                    save_dict = {
                        'video': video,
                        'motion_target': motion_target,
                        'source_latent': target,
                        'loss_mask': loss_mask,
                        'student_preds': student_preds
                    }
                    torch.save(save_dict, SV_PATH)
                    
            metrics = {'loss': loss}

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(epoch, i_batch + 1, metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                if args.model == 'affinity':
                    torch.save(model.state_dict(), PATH)
                elif args.model == 'motion':
                    torch.save(student_model.state_dict(), PATH)

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
                if args.stage in ['sintel']:
                    model.module.freeze_bn()

            total_steps += 1
            t2 = time.time()
            if args.time_it:
                print("step time", i_batch, t2-t1)

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    if args.model == 'affinity':
        torch.save(model.state_dict(), PATH)
    elif args.model == 'motion':
        torch.save(student_model.state_dict(), PATH)

    return PATH

def get_args(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_it', action='store_true')
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default="chairs", help="determines which dataset to use for training")
    parser.add_argument('--video_length', type=int, default=5)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default='/data2/honglinc/')
    parser.add_argument('--dataset_names', type=str, nargs='+')
    parser.add_argument('--train_split', type=str, default='all')
    parser.add_argument('--flow_gap', type=int, default=1)
    parser.add_argument('--filepattern', type=str, default="*", help="which files to train on tdw")
    parser.add_argument('--test_filepattern', type=str, default="*9", help="which files to val on tdw")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--gate_ckpt', help="restore checkpoint")    
    parser.add_argument('--teacher_config', type=str, default=None)
    parser.add_argument('--student_config', type=str, default=None)        
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--small_teacher', action='store_true', help='use small model')            
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--val_freq', type=int, default=5000, help='validation and checkpoint frequency')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--save_students', action='store_true')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--teacher_iters', type=int, default=24)        
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--gate_stride', type=int, default=2)
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
    parser.add_argument('--gate_model', type=str, default='gate', help='Gate Model class')    
    parser.add_argument('--predict_mask', action='store_true', help='Whether to predict a thingness mask')
    parser.add_argument('--bootstrap', action='store_true', help='whether to bootstrap')
    parser.add_argument('--teacher_ckpt', help='checkpoint for a pretrained RAFT. If None, use GT')
    parser.add_argument('--gate_iters', type=int, default=24)
    parser.add_argument('--motion_iters', type=int, default=12)
    parser.add_argument('--boundary_iters', type=int, default=12)
    parser.add_argument('--flow_iters', type=int, default=12)
    parser.add_argument('--motion_ckpt', help='checkpoint for a pretrained motion model')
    parser.add_argument('--boundary_ckpt', help='checkpoint for a pretrained boundary model')
    parser.add_argument('--flow_ckpt', help='checkpoint for a pretrained boundary model')
    parser.add_argument('--static_ckpt', help='checkpoint for a pretrained eisen model')

    # BBNet teacher params
    parser.add_argument('--static_resolution', type=int, default=4)
    parser.add_argument('--dynamic_resolution', type=int, default=3)
    parser.add_argument('--affinity_kernel_size', type=int, default=None)
    parser.add_argument('--motion_mask_target', action='store_true')

    # motion propagation
    parser.add_argument('--diffusion_target', action='store_true')
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--rgb_flow', action='store_true')
    parser.add_argument('--boundary_flow', action='store_true')
    parser.add_argument('--separate_boundary_models', action='store_true')
    parser.add_argument('--zscore_target', action='store_true')
    parser.add_argument('--downsample_factor', type=int, default=2)
    parser.add_argument('--teacher_downsample_factor', type=int, default=1)
    parser.add_argument('--patch_radius', type=int, default=1)
    parser.add_argument('--warp_radius', type=int, default=5)
    parser.add_argument('--probs_per_frame', action='store_true')
    parser.add_argument('--positive_samples_only', action='store_true')            
    parser.add_argument('--motion_thresh', type=float, default=None)
    parser.add_argument('--boundary_thresh', type=float, default=None)
    parser.add_argument('--target_thresh', type=float, default=0.75)
    parser.add_argument('--pathway_thresh', type=float, default=0.75)    
    parser.add_argument('--temporal_thresh', type=float, default=0)    
    parser.add_argument('--pixel_thresh', type=int, default=None)
    parser.add_argument('--positive_thresh', type=float, default=0.4)
    parser.add_argument('--negative_thresh', type=float, default=0.1)
    parser.add_argument('--affinity_radius', type=int, default=1)
    parser.add_argument('--affinity_loss_type', type=str, default='kl_div')
    parser.add_argument('--static_input', action='store_true')
    parser.add_argument('--affinity_nonlinearity', type=str, default='softmax_max')
    parser.add_argument('--num_propagation_iters', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--integration_window', type=int, default=5)    
    parser.add_argument('--num_sample_points', type=int, default=2**14)
    parser.add_argument('--predict_every', type=int, default=5)
    parser.add_argument('--binarize_motion', action='store_true')
    parser.add_argument('--use_motion_loss', action='store_true')
    parser.add_argument('--loss_scale', type=float, default=1.0)

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

    path = Path(load_path) if load_path else None

    def _get_model_class(name):
        cls = None
        if 'bootraft' in name:
            cls = BootRaft
        elif 'raft' in name:
            cls = RAFT
        elif ('thing' in name) or ('occlusion' in name):
            cls = ThingsClassifier
        elif 'centroid' in name:
            cls = CentroidRegressor
        elif 'motion' in name:
            cls = MotionClassifier
        elif 'prop' in name:
            cls = MotionPropagator
        elif 'boundary' in name:
            cls = BoundaryClassifier
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
