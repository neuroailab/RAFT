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
from torchvision import transforms

from torch.utils.data import DataLoader
from raft import (RAFT,
                  ThingsClassifier,
                  CentroidRegressor,
                  MotionClassifier,
                  BoundaryClassifier,
                  MotionPropagator)

from bootraft import (BootRaft,
                      CentroidMaskTarget,
                      ForegroundMaskTarget,
                      IsMovingTarget,
                      DiffusionTarget)
import teachers

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

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, min_flow=0.5, pos_weight=1.0, pixel_thresh=None):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid is None:
        valid = (mag < max_flow)
    else:
        valid = (valid >= 0.5) & (mag < max_flow)
        valid = valid.float()
    num_px = valid.sum((-2,-1)).clamp(min=1)

    if list(flow_gt.shape[-2:]) != list(flow_preds[-1].shape[-2:]):
        _ds = lambda x: F.avg_pool2d(
            x,
            args.downsample_factor * args.teacher_downsample_factor,
            stride=args.downsample_factor * args.teacher_downsample_factor)
    else:
        _ds = lambda x: x

    if flow_preds[-1].shape[-3] == 1:
        flow_gt = (mag[:,None] > min_flow).float()

        pos_weight = torch.tensor([pos_weight], device=flow_gt.device)
        loss_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        loss_fn = lambda logits, labels: loss_cls(_ds(logits), labels)
    else:
        loss_fn = lambda logits, labels: (_ds(logits) - labels).abs()
        assert flow_preds[-1].shape[-3] == 2, flow_preds[-1].shape


    # print("num foreground px", flow_gt[0].sum())
    # print("total pixels", np.prod(flow_gt.shape[-2:]))

    if pixel_thresh is not None:
        print("pos px", flow_gt.sum((1,2,3)))
        gt_weight = (flow_gt.sum((1,2,3), True) > pixel_thresh).float()
        print("gt weight", gt_weight[:,0,0,0])
    else:
        gt_weight = 1.0


    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = loss_fn(flow_preds[i], flow_gt) * gt_weight
        flow_loss += ((i_weight * valid * i_loss).sum((-2,-1)) / num_px).mean()

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

def boundary_loss(boundary_preds,
                  boundary_target,
                  valid,
                  gamma=0.8,
                  boundary_scale=1.0,
                  orientation_scale=1.0,
                  pixel_thresh=None,
                  **kwargs):

    n_predictions = len(boundary_preds)
    b_loss = c_loss = loss = 0.0

    # break up boundary_target
    if boundary_target.shape[1] == 3:
        b_target, c_target = boundary_target.split([1,2], 1)
    else:
        b_target, c_target, c_target_discrete = boundary_target.split([1,2,8], 1)
    num_px = b_target.sum(dim=(-3,-2,-1)).clamp(min=1.)

    if pixel_thresh is not None:
        print("pos px", b_target.sum((1,2,3)))
        gt_weight = (b_target.sum((1,2,3), True) > pixel_thresh).float()
        print("gt weight", gt_weight[:,0,0,0])
    else:
        gt_weight = 1.0

    def _split_preds(x):
        dim = x.shape[-3]
        if dim == 3:
            return x.split([1,2], -3)
        elif dim == 9:
            c1, b, c2 = x.split([4,1,4], -3)
            return b, torch.cat([c1, c2], -3)

    b_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    if boundary_preds[-1].shape[1] == 3:
        c_loss_fn = lambda logits, labels: (logits - labels).abs().sum(1)
    else:
        c_loss_fn = nn.CrossEntropyLoss(reduction='none')
        c_target = c_target_discrete.argmax(1)

    ds = args.downsample_factor * args.teacher_downsample_factor
    if list(b_target.shape[-2:]) != list(boundary_preds[-1].shape[-2:]):
        _ds = lambda x: F.avg_pool2d(x, ds, stride=ds)
    else:
        _ds = lambda x:x

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        b_pred, c_pred = _split_preds(_ds(boundary_preds[i]))
        i_b_loss = (b_loss_fn(b_pred, b_target) * gt_weight).mean()
        i_c_loss = (c_loss_fn(c_pred, c_target)[:,None] * b_target * gt_weight).sum((-3,-2,-1))
        i_c_loss = (i_c_loss / num_px).mean()

        b_loss += i_b_loss * i_weight
        c_loss += i_c_loss * i_weight
        i_loss = i_b_loss * boundary_scale +\
                 i_c_loss * orientation_scale
        loss += i_loss * i_weight

    metrics = {
        'loss': loss,
        'b_loss': b_loss,
        'c_loss': c_loss
    }

    return loss, metrics


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

    if args.model.lower() == 'bootraft':
        model_cls = BootRaft
        print("used BootRaft")
    elif args.model.lower() == 'thingness' or args.model.lower() == 'occlusion':
        model_cls = ThingsClassifier
        print("used ThingnessClassifier")
    elif args.model.lower() == 'centroid' or args.model.lower() == 'motion_centroid':
        model_cls = CentroidRegressor
        print("used CentroidRegressor")
    elif args.model.lower() == 'motion':
        model_cls = MotionClassifier
        print("used MotionClassifier")
    elif args.model.lower() == 'boundary':
        model_cls = BoundaryClassifier
        print("used BoundaryClassifier")
    elif args.model.lower() == 'flow':
        model_cls = RAFT
        print("used RAFT")
    model = nn.DataParallel(model_cls(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    print("Using ground truth? %s" % args.supervised)

    if args.restore_ckpt is not None:
        did_load = model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print(did_load, type(model.module).__name__, args.restore_ckpt)

    model.cuda()
    model.train()

    if args.stage in ['sintel']:
        model.module.freeze_bn()

    ## load a teacher model
    selfsup = False

    ## if training on a dataset without gt flow, we __must__ self-supervise
    if args.stage in SELFSUP_DATASETS:
        selfsup = True
        assert args.teacher_ckpt is not None, "You're training on %s that has no gt flow! Pass a teacher checkpoint" % args.stage

    if args.teacher_ckpt is not None:
        selfsup = True
        # _small = args.small
        # args.small = False
        # teacher = nn.DataParallel(RAFT(args), device_ids=args.gpus)
        # did_load = teacher.load_state_dict(torch.load(args.teacher_ckpt), strict=False)
        # args.small = _small
        teacher = load_model(
            load_path=args.teacher_ckpt,
            small=('small' in str(args.teacher_ckpt)),
            cuda=True, train=False,
            gpus=args.gpus
        )

        print("TEACHER")
        teacher.eval()
        teacher.module.freeze_bn()
    else:
        teacher = None

    ## if training on Thingness or Centroids on DAVIS, we construct a foreground mask out of the flow
    target_net = None
    if (args.stage.lower() == 'davis') and (args.model.lower() in ['thingness', 'centroid']):
        target_net = nn.DataParallel(ForegroundMaskTarget(mask_input=True, get_connected_component=True), device_ids=args.gpus).cuda()
    elif (args.model.lower() in ['occlusion',
                                 'motion',
                                 'motion_centroid',
                                 'thingness',
                                 'boundary'
    ]) and not args.supervised and not args.bootstrap:
        assert args.no_aug, "Can't use data augmentation with motion target"
        if args.diffusion_target:
            target_net = nn.DataParallel(DiffusionTarget(
                patch_radius=args.patch_radius,
                downsample_factor=args.downsample_factor,
                num_samples=args.num_samples,
                num_points=args.num_sample_points,
                confidence_thresh=args.target_thresh,
                num_iters=args.num_propagation_iters,
                noise_thresh=args.motion_thresh,
                energy_prior_gate=(teacher is None)
            )).cuda()
            print("Diffusion Target")
        else:
            target_net = nn.DataParallel(IsMovingTarget(
                patch_radius=args.patch_radius,
                thresh=args.motion_thresh,
                zscore_target=args.zscore_target,
                normalize_error=False,
                normalize_features=False,
                get_errors=args.use_motion_loss,
                size=None), device_ids=args.gpus).cuda()
            print("IsMoving Target")

        ## teacher just stacks the images
        selfsup = True
        def get_video(img1, img2, img0, **kwargs):
            assert img1.dtype == img2.dtype == img0.dtype == torch.float32
            return (None, torch.stack([img0, img1, img2], 1) / 255.0)
    elif args.model.lower() == 'flow' and not args.bootstrap:
        target_net = nn.DataParallel(teachers.FuturePredictionTeacher(
            downsample_factor=args.teacher_downsample_factor,
            target_motion_thresh=args.motion_thresh,
            target_boundary_thresh=args.boundary_thresh,
            concat_rgb_features=(args.rgb_flow),
            concat_boundary_features=(args.boundary_flow),
            concat_orientation_features=(args.boundary_flow),
            concat_fire_features=False,
            motion_path=args.motion_ckpt,
            boundary_path=args.boundary_ckpt,
            parse_paths=True
        ), device_ids=args.gpus).cuda()
        target_net.eval()
        print("FuturePredictionTarget")
        selfsup = True
        def get_video(img1, img2, img0, **kwargs):
            assert img1.dtype == img2.dtype == img0.dtype == torch.float32
            return (None, torch.stack([img0, img1, img2], 1) / 255.0)

    elif args.model.lower() in ['motion', 'boundary', 'flow'] and args.bootstrap:
        assert args.flow_ckpt is not None

        target_net = nn.DataParallel(teachers.MotionToStaticTeacher(
            student_model_type=args.model.lower(),
            build_flow_target=True,
            downsample_factor=args.teacher_downsample_factor,
            motion_path=args.motion_ckpt,
            boundary_path=args.boundary_ckpt,
            flow_path=args.flow_ckpt,
            parse_paths=True,
            target_motion_thresh=args.motion_thresh,
            target_boundary_thresh=args.boundary_thresh,
        ), device_ids=args.gpus).cuda()
        target_net.eval()
        print(type(target_net.module).__name__)
        selfsup = True
        def get_video(img1, img2, img0, **kwargs):
            assert img1.dtype == img2.dtype == img0.dtype == torch.float32
            return (None, torch.stack([img0, img1, img2], 1) / 255.0)

    train_loader, epoch_size = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

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
            if selfsup:
                image1, image2 = [x.cuda() for x in data_blob[:2]]
                valid = None
                if args.model.lower() in ['motion',
                                          'occlusion',
                                          'thingness',
                                          'boundary',
                                          'flow'
                ]:
                    image0 = data_blob[2].cuda()

            elif len(data_blob) == 3:
                image1, image2, flow = [x.cuda() for x in data_blob]
                valid = None
            else:
                image1, image2, flow, valid = [x.cuda() for x in data_blob]


            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            ## get the self-supervision
            _ds = lambda x: x
            if args.teacher_downsample_factor > 1:
                H,W = image1.shape[-2:]
                stride = args.teacher_downsample_factor
                _ds = transforms.Resize([H // stride, W // stride])
            if selfsup and (not args.bootstrap) and args.model.lower() in ['motion',
                                                                           'occlusion',
                                                                           'thingness',
                                                                           'boundary']:

                if teacher is None:
                    _, flow = get_video(_ds(image1), _ds(image2), _ds(image0),
                                        iters=args.teacher_iters, test_mode=True)
                else:
                    _, video = get_video(_ds(image1), _ds(image2), _ds(image0),
                                        iters=args.teacher_iters, test_mode=True)
                    _, prior = teacher(image1, image2, iters=args.teacher_iters, test_mode=True)
                    assert prior.shape[1] == 1, prior.shape
                    prior = prior.detach().sigmoid()
                    if list(prior.shape[-2:]) != list(video.shape[-2:]):
                        prior = _ds(prior)

                    flow = [video, prior]

            elif selfsup and args.bootstrap and (args.model.lower() in ['flow', 'motion', 'boundary']):
                assert target_net is not None
                flow = [image1, image2]
            elif selfsup and args.model.lower() == 'flow':
                assert target_net is not None
                flow = [image1, image2]
            elif selfsup:
                _, flow = teacher(_ds(image1), _ds(image2), iters=args.teacher_iters, test_mode=True)
                # print("model", args.model, "target", flow.shape, flow.amin(), flow.amax())

            ## process the gt
            if target_net is not None:
                if not isinstance(flow, (list, tuple)):
                    flow = [flow]
                flow = target_net(
                    *flow,
                    motion_iters=args.motion_iters,
                    boundary_iters=args.boundary_iters,
                    flow_iters=args.flow_iters
                )

                if isinstance(flow, (tuple, list)):
                    if 'combined' in args.orientation_type:
                        flow = torch.cat([flow[0], flow[1]], 1)
                    elif args.model.lower() == 'boundary':
                        flow = flow[1]
                    elif args.model.lower() == 'flow' and args.bootstrap:
                        flow = flow
                    elif args.model.lower() == 'flow' and args.boundary_flow:
                        flow, valid = flow[0], flow[2]
                        print("num px", valid.sum(), np.prod(list(valid.shape)))
                    else:
                        flow = flow[0]

                print("TARGET SHAPE", flow.shape, args.model.lower())
                if len(flow.shape) == 5:
                    flow = flow.squeeze(1)

                # # save a tensor
                # PATH = 'checkpoints/dtarget_%s.pth' % args.name
                # torch.save({
                #     'prior': prior,
                #     'video': video,
                #     'target': flow
                # }, PATH)


            if args.model.lower() in ['centroid', 'motion_centroid']:
                loss, metrics = centroid_loss(flow_predictions, flow, valid, args.gamma, scale_to_pixels=args.scale_centroids, pixel_thresh=args.pixel_thresh)
            elif args.use_motion_loss and args.model.lower() == 'motion':
                loss, metrics = motion_loss(flow_predictions, flow, valid, args.gamma, loss_scale=args.loss_scale)
            elif args.model.lower() == 'boundary':
                if args.orientation_type == 'combined':
                    loss, metrics = boundary_loss(
                        [f[:,1:] for f in flow_predictions], flow[:,1:],
                        valid, args.gamma, pixel_thresh=args.pixel_thresh)
                    loss_m, metrics_m = sequence_loss(
                        [f[:,:1] for f in flow_predictions], flow[:,:1],
                        valid, args.gamma, pos_weight=args.pos_weight,
                        pixel_thresh=args.pixel_thresh)
                    loss = loss + loss_m
                    metrics['motion_loss'] = metrics_m['loss']
                    metrics['loss'] += metrics_m['loss']
                else:
                    loss, metrics = boundary_loss(flow_predictions, flow, valid, args.gamma, pixel_thresh=args.pixel_thresh)
            else:
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma, pos_weight=args.pos_weight, pixel_thresh=args.pixel_thresh)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(epoch, i_batch + 1, metrics)

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
                if args.stage in ['sintel']:
                    model.module.freeze_bn()

            total_steps += 1
            t2 = time.time()
            print("step time", i_batch, t2-t1)

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
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
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
    parser.add_argument('--supervised', action='store_true', help='whether to supervise')
    parser.add_argument('--bootstrap', action='store_true', help='whether to bootstrap')
    parser.add_argument('--teacher_ckpt', help='checkpoint for a pretrained RAFT. If None, use GT')
    parser.add_argument('--teacher_iters', type=int, default=24)
    parser.add_argument('--motion_iters', type=int, default=12)
    parser.add_argument('--boundary_iters', type=int, default=12)
    parser.add_argument('--flow_iters', type=int, default=12)
    parser.add_argument('--motion_ckpt', help='checkpoint for a pretrained motion model')
    parser.add_argument('--boundary_ckpt', help='checkpoint for a pretrained boundary model')
    parser.add_argument('--flow_ckpt', help='checkpoint for a pretrained boundary model')
    parser.add_argument('--features_ckpt', help='checkpoint for a pretrained features model')

    # motion propagation
    parser.add_argument('--diffusion_target', action='store_true')
    parser.add_argument('--orientation_type', default='classification')
    parser.add_argument('--rgb_flow', action='store_true')
    parser.add_argument('--boundary_flow', action='store_true')
    parser.add_argument('--separate_boundary_models', action='store_true')
    parser.add_argument('--zscore_target', action='store_true')
    parser.add_argument('--downsample_factor', type=int, default=1)
    parser.add_argument('--teacher_downsample_factor', type=int, default=1)
    parser.add_argument('--patch_radius', type=int, default=0)
    parser.add_argument('--motion_thresh', type=float, default=None)
    parser.add_argument('--boundary_thresh', type=float, default=None)
    parser.add_argument('--target_thresh', type=float, default=0.75)
    parser.add_argument('--pixel_thresh', type=int, default=None)
    parser.add_argument('--positive_thresh', type=float, default=0.4)
    parser.add_argument('--negative_thresh', type=float, default=0.1)
    parser.add_argument('--affinity_radius', type=int, default=1)
    parser.add_argument('--affinity_radius_inference', type=int, default=1)
    parser.add_argument('--static_affinities', action='store_true')
    parser.add_argument('--static_input', action='store_true')
    parser.add_argument('--affinity_nonlinearity', type=str, default='softmax')
    parser.add_argument('--num_propagation_iters', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=8)
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

def load_motion_teacher(args):

    ## get a path for the motion classifier
    motion_path = args.motion_ckpt
    if motion_path is not None:
        motion_model = load_model(motion_path,
                                  small=True,
                                  cuda=True, train=False
        )
    else:
        motion_model = None

    ## get a path for the features
    features_path = args.features_ckpt
    if features_path is not None:
        features_model = load_model(features_path, small=False, cuda=True, train=False)
    else:
        features_model = None

    ## the teacher model
    def teacher(img1, img2, **kwargs):

        if motion_model is not None:
            _, motion = motion_model(img1, img2, **kwargs)
            motion = (torch.sigmoid(motion) > args.motion_thresh).float()
        else:
            motion = torch.ones_like(img).mean(-3, True)

        if features_model is not None:
            _, feats1 = features_model(img1, img1, **kwargs)
            _, feats2 = features_model(img2, img2, **kwargs)
            feats = torch.stack([feats1, feats2], 1)
        else:
            feats = torch.stack([img1, img2], 1) / 255.0

        target_feats = motion[:,None] * feats
        return (None, target_feats)

    return teacher

if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
