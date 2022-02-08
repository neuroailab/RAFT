import os, sys
sys.path.append('./core')

import h5py
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
from raft import RAFT
from bootraft import BootRaft

from argparse import ArgumentParser

def get_args(cmd=None):
    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile", type=str, help="name of outfile")
    parser.add_argument("--model", type=str, default="RAFT", help="model class")
    parser.add_argument("--checkpoint", help="checkpoint to load")

    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--static_coords', action='store_true', help='use mixed precision')

    ## dataset
    parser.add_argument("--dataset", type=str, default="tdw", help="which dataset to filter")
    parser.add_argument("--num_files", type=int, default=-1, help="how many files to search over")

    if cmd is None:
        args = parser.parse_args()
        print(args)
    else:
        args = parser.parse_args(cmd)
    return args

class DeltaImages(nn.Module):
    def __init__(self, args, thresh=0.025):
        super().__init__()
        self.args = args
        self.thresh = thresh
    def forward(self, img1, img2, **kwargs):
        video = torch.stack([img1, img2], 1)
        B = video.shape[0]
        size = video.shape[-2:]

        if video.dtype == torch.uint8:
            video = video.float() / 255.
        if video.shape[1] == 1:
            return torch.zeros([B,1,1,size[0],size[1]]).float().to(video.device)
        delta = (video[:,:-1] - video[:,1:]).square().sum(-3, True).sqrt()
        if self.thresh is not None:
            return (delta > self.thresh).float()
        return delta

def load_model(args):

    if args.model.lower() == 'raft':
        model_cls = RAFT
    elif args.model.lower() == 'bootraft':
        model_cls = BootRaft
    elif args.model.lower() == 'delta':
        model_cls = DeltaImages
    else:
        raise ValueError("%s is not a valid teacher model class" % args.model)

    ## instantiate the model
    model = nn.DataParallel(model_cls(args), device_ids=args.gpus)

    if args.checkpoint is not None:
        res = model.load_state_dict(torch.load(args.checkpoint), strict=False)
        print(res)

    model.cuda()
    model.eval()

    return model

def get_dataset(args):

    if args.dataset == 'tdw':
        dataset = datasets.TdwFlowDataset(
            aug_params=None,
            split='training')
    else:
        raise ValueError("%s is not a recognized dataset" % args.dataset)

    assert hasattr(dataset, 'get_video'), "Dataset object must have a method 'get_image_pair'"
    assert hasattr(dataset, 'files'), "Dataset must have a list of files"

    print("Searching over %d files" % len(dataset.files))
    return dataset

class FrameScore(nn.Module):
    frame_offset = 0
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def score_prediction(self, pred):
        raise NotImplementedError("You need to implement a scoring method")

    def forward(self, *args, **kwargs):

        pred = self.model(*args, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        score = self.score_prediction(pred)
        return score.item()

class TotalEnergyScore(FrameScore):
    frame_offset = 0
    def score_prediction(self, pred):
        return pred.square().sum(-3).sqrt().mean()

class TrainingMovieFinder(object):
    """a class for reading in movies from a Torch dataset and filtering according to motion"""
    def __init__(self,
                 model: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 video_length: int = 3,
                 num_files: int = 3,
                 score_config=(TotalEnergyScore, {}),
                 model_call_kwargs = {'iters': 6, 'test_mode': True}
    ):
        self.model = model
        self.model_call_kwargs = model_call_kwargs
        self.dataset = dataset

        self.infiles = self.dataset.files
        if num_files is not None:
            self.infiles = self.infiles[:num_files]
        self.num_files = len(self.infiles)

        self.video_length = video_length

        ## set the score fn
        self.set_score_fn(score_config)

    def set_score_fn(self, config):
        if hasattr(config, 'keys'):
            func = config.pop('func', TotalMotionScore)
            self.score_fn = func(model=self.model, **config)
        elif hasattr(config, '__len__'):
            if len(config) == 1:
                self.score_fn = config(model=self.model)
            elif len(config) == 2:
                self.score_fn = config[0](model=self.model, **config[1])
            else:
                raise ValueError()
        else:
            raise ValueError()

        self._score_offset = getattr(self.score_fn, 'frame_offset', 0)

    def _score_video(self, video):
        n_frames = len(video)
        to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
        for i in range(n_frames - 1):
            img1, img2 = to_tensor(video[i]), to_tensor(video[i+1])
            score = self.score_fn(img1, img2, **self.model_call_kwargs)
            print(score)
        return score

    def filter_video(self, file_idx):

        file_idx = file_idx % self.num_files
        filename = self.infiles[file_idx]
        f = h5py.File(filename, 'r')

        frame, video = 0, []
        scores = {}
        while (video is not None):
            video = self.dataset.get_video(f, frame, self.video_length)
            if video is not None:
                scores[frame + self._score_offset] = self._score_video(video)
            frame += 1

        f.close()
        print(scores)
        return scores

## TODO
## read in a file
## for each movie of length (eval_len):
## read movie into memory
## pass image pairs through model to get preds
## compute metric on preds
## store data for this video
## apply threshold and/or max
## append to json outfile

if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('datasets/supervision_frames'):
        os.mkdir('datasets/supervision_frames')

    model = load_model(args)
    dataset = get_dataset(args)
    print(dataset.files[:3])

    finder = TrainingMovieFinder(model, dataset)
    for i in range(3):
        scores = finder.filter_video(i)
        keys = sorted(scores.keys())
        print("argmax", np.argmax(np.array([scores[k] for k in keys])))
