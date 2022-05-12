import os,sys,json
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

from tqdm import tqdm

METRIC_NAMES = ['ari', 'fari', 'miou', 'recall50']
DEFAULT_BBNET_PATHS = {
    'motion_path': 'checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth',
    'boundary_path': 'checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth',
    'flow_path': 'checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth',
    'dynamic_path': None,
    'static_path': None,
    'centroid_path': None
}

def get_args(cmd=None):
    parser = argparse.ArgumentParser()

    ## BBNet component paths
    parser.add_argument('--motion_path', type=str, default=DEFAULT_BBNET_PATHS['motion_path'])
    parser.add_argument('--boundary_path', type=str, default=DEFAULT_BBNET_PATHS['boundary_path'])
    parser.add_argument('--flow_path', type=str, default=DEFAULT_BBNET_PATHS['flow_path'])
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
    parser.add_argument('--flow_iters', type=int, default=24, help="How many iters to run flow network")
    parser.add_argument('--mask_with_motion', action='store_true',
                        help="Whether to motion mask plateau map")

    ## running on gpu
    parser.add_argument('--gpus', type=str, default='0')

    ## dataset
    parser.add_argument('--dataset', type=str, default='movi_d')
    parser.add_argument('--dataset_dir', type=str, default='/data2/honglinc/')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seq_length', type=int, default=6, help="Length of video to evaluate on")
    parser.add_argument('--start_frame', type=int)
    parser.add_argument('--test_size', type=int, nargs='+')
    parser.add_argument('--ex_start', type=int, default=0, help="Which example to start at")
    parser.add_argument('--num_examples', type=int, default=1, help="How many examples to evaluate")

    ## saving results
    parser.add_argument('--out_dir', type=str, default='results/bbnet')
    parser.add_argument('--outfile', type=str)
    

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

def get_ari(pred, gt, ignore_background=True):

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
        ignore_background=ignore_background
    ).mean().cpu().item()
    return ari

def save_results(args, results):
    import pickle
    outfile = args.outfile
    if outfile[-4:] != '.pkl':
        outfile += '.pkl'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    outpath = os.path.join(args.out_dir, outfile)
    with open(outpath, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def aggregate_results(results):

    examples = sorted(results.keys())
    if len(examples) == 0:
        return None

    metric_names = results[examples[0]].keys()
    agg = {nm:[] for nm in metric_names}

    for ex in examples:
        for nm in metric_names:
            agg[nm].append(results[ex][nm])

    for nm in metric_names:
        if nm in METRIC_NAMES:
            agg[nm+'_std'] = np.nanstd(np.array(agg[nm]))            
            agg[nm] = np.nanmean(np.array(agg[nm]))
            
    return agg
    
def test(args):

    bbnet = load_bbnet(args)
    dataset = fetch_dataset(args)
    print("num parameters in model --- %d" % sum(p.numel() for p in bbnet.parameters()))

    ## loop over examples
    ## TODO batching and parallelization across gpus
    ex_start = args.ex_start
    num_examples = args.num_examples

    ## Since we can't randomly access examples with a tensorflow_dataset, just burn through
    for ex in range(ex_start):
        _ = dataset[ex]

    ## actual eval loop
    results = OrderedDict()
    for ex in tqdm(range(ex_start, ex_start + num_examples)):
        data = dataset[ex]
        pred_segments = bbnet(
            video=data['images'][None].cuda(),
            boot_params={'bootstrap': args.bootstrap, 'flow_iters': args.flow_iters},
            static_params={'to_image': True, 'local_window_size': None},
            mask_with_motion=args.mask_with_motion
        )
        print("pred segments", pred_segments.shape)
        gt_segments = data['objects'][None,:,0].long().cuda()


        results[ex] = {
            'example': ex,
            'video_name': dataset.meta['video_name'],
            'fari': get_ari(pred_segments, gt_segments, ignore_background=True),
            'ari': get_ari(pred_segments, gt_segments, ignore_background=False)
        }
        print(ex, results[ex])

    save_results(args, results)
    agg_results = aggregate_results(results)
    print(agg_results)
    

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    args = get_args()
    test(args)
