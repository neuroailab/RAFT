import os,sys
import argparse
from collections import OrderedDict
sys.path.append('core')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import teachers
import dorsalventral.data.movi as movi_dataset
import dorsalventral.models.fire_propagation as fprop
import dorsalventral.evaluation.object_metrics as metrics
import kornia

def get_args(cmd=None):
    parser = argparse.ArgumentParser()

    ## BBNet component paths
    parser.add_argument('--motion_path', type=str)
    parser.add_argument('--boundary_path', type=str)
    parser.add_argument('--flow_path', type=str)
    parser.add_argument('--dynamic_path', type=str)
    parser.add_argument('--static_path', type=str)
    parser.add_argument('--centroid_path', type=str)

    ## BBNet parameters
    parser.add_argument('--static_res', type=int, default=4, help="height/width resolution in plateau")
    parser.add_argument('--dynamic_res', type=int, default=3, help="flowH/flowW resolution in plateau")

    ## spatial and temporal sampling
    parser.add_argument('--stride', type=int, default=4, help="How much to spatially downsample")
    parser.add_argument('--tgroup', type=int, default=4, help="Length of sliding window for grouping")
    parser.add_argument('--ttrack', type=int, default=2, help="Step size for tracking")
    parser.add_argument('--affinity_size', type=int, nargs='+', help="Specify affinity size for EISEN")

    ## call params
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--flow_iters', type=int, default=12, help="How many iters to run flow network")
    parser.add_argument('--mask_with_motion', action='store_true',
                        help="Whether to motion mask plateau map")

    ## running on gpu
    parser.add_argument('--gpus', type=str, default='0')

    ## dataset
    parser.add_argument('--dataset', type=str, default='movi_d')
    parser.add_argument('--dataset_dir', type=str, default='/data5/dbear/movi_datasets')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seq_length', type=int, default=6, help="Length of video to evaluate on")
    parser.add_argument('--start_frame', type=int)
    parser.add_argument('--test_size', type=int, nargs='+')

    if cmd is None:
        args = parser.parse_args()
        print(args)
    else:
        args = arser.parse_args(cmd)
    return args

def load_bbnet(args):

    stride = args.stride
    eisen_params = {
        'stem_pool': (stride == 4),
        'affinity_res': [256//stride, 256//stride]
    }

    bbnet = teachers.BipartiteBootNet(
        student_model_type='eisen',
        boot_paths={
            'motion_path': args.motion_path,
            'boundary_path': args.boundary_path,
            'flow_path': args.flow_path
        },
        static_path=args.static_path,
        dynamic_path=args.dynamic_path,
        centroid_path=args.centroid_path,
        static_params=eisen_params,
        downsample_factor=args.stride,
        grouping_window=args.tgroup,
        tracking_step_size=args.ttrack,
        static_resolution=args.static_res,
        dynamic_resolution=args.dynamic_res
    ).cuda().eval()

    return bbnet

def fetch_dataset(args):

    data_dir = os.path.join(args.dataset_dir, args.dataset, '256x256', '1.0.0')
    start_frame = args.start_frame or (12 - (args.seq_length//2))
    dataset = movi_dataset.MoviDataset(
        dataset_dir=data_dir,
        split=args.split,
        sequence_length=args.seq_length,
        min_start_frame=start_frame,
        max_start_frame=start_frame,
        is_test=True,
        passes=["images", "objects", "flow"]
    )
    print("dataset has %d examples" % len(dataset))
    return dataset

def get_ari(pred, gt):

    size_pred = list(pred.shape[-2:])
    size_gt = list(gt.shape[-2:])
    if args.test_size is None:
        pred = transforms.Resize(size_gt, interpolation=transforms.InterpolationMode.NEAREST)(pred)
    else:
        resize = transforms.Resize(args.test_size, interpolation=transforms.InterpolationMode.NEAREST)
        pred, gt = resize(pred), resize(gt)

    ari = metrics.adjusted_rand_index(
        pred_ids=(pred + 1).long(),
        true_ids=gt,
        num_instances_pred=(pred + 1).amax().item() + 1,
        num_instances_true=gt.amax().item() + 1,
        ignore_background=True
    )
    return ari

def test(args):

    bbnet = load_bbnet(args)
    dataset = fetch_dataset(args)
    print("num parameters in model --- %d" % sum(p.numel() for p in bbnet.parameters()))

    ex = 0
    data = dataset[ex]
    pred_segments = bbnet(
        video=data['images'][None].cuda(),
        boot_params={'bootstrap': args.bootstrap, 'flow_iters': args.flow_iters},
        static_params={'to_image': True, 'local_window_size': None},
        mask_with_motion=args.mask_with_motion
    )
    print("pred segments", pred_segments.shape)
    gt_segments = data['objects'][None,:,0].long().cuda()
    ari = get_ari(pred_segments, gt_segments)

    results = OrderedDict()
    print("ari", ari)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    args = get_args()
    test(args)
