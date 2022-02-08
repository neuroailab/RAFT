import os, sys
sys.path.append('./core')

import h5py
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import datasets
from raft import RAFT
from bootraft import BootRaft
from kmeans import KMeans

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

    ## scoring
    parser.add_argument("--score_func", default="total_energy", help="Which score function to use")

    ## main
    parser.add_argument("--overwrite", action="store_true", help="overwrite the file if it exists")
    parser.add_argument("--verbose", action="store_true")

    if cmd is None:
        args = parser.parse_args()
        print(args)
    else:
        args = parser.parse_args(cmd)
    return args

class DeltaImages(nn.Module):
    num_input_frames = 2
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
    defaults = {}
    def score_prediction(self, pred):
        return pred.square().sum(-3).sqrt().mean()

class MaximumEnergyScore(FrameScore):
    frame_offset = 0
    defaults = {}
    def score_prediction(self, pred):
        return pred.square().sum(-3).sqrt().amax()

class ThresholdedEnergyScore(FrameScore):
    frame_offset = 0
    defaults = {'thresh': 0.5}
    def __init__(self, thresh=0.5, *args, **kwargs):
        self.thresh = thresh
        super().__init__(*args, **kwargs)
    def score_prediction(self, pred):
        energy = pred.square().sum(-3).sqrt()
        if self.thresh:
            energy = (energy > self.thresh).float()
        return energy.mean()

class VideoSegmentationModel(nn.Module):
    num_input_frames = 2
    def __init__(self, model, num_input_frames=None, **kwargs):
        super().__init__()
        self.model = model
        self.model.cuda()
        self.model.eval()

        if num_input_frames is not None:
            self.num_input_frames = num_input_frames

    def forward(self, *args, **kwargs):
        images = args[:self.num_input_frames]
        images = torch.stack(images, 1) # [B,num_frames,C,H,W]
        images = images.to(torch.uint8)
        assert len(images.shape) == 5, images.shape

        segments = self.model(images)
        assert len(segments.shape) == 3, segments.shape
        assert segments.dtype == torch.long, segments.dtype
        return segments

class ExplainedAwayMotion(FrameScore):
    frame_offset = 0
    defaults = {'thresh': 0.5}
    def __init__(self,
                 model,
                 segment_model,
                 thresh=0.5,
                 **kwargs):
        super().__init__(model=model)
        self.segment_model = segment_model
        self.segment_model.cuda()
        self.segment_model.eval()
        self.thresh = thresh

    def score_prediction(self, pred_motion, pred_segments):
        ## get the motion segments by thresholding

        ## get the static segments

        ## find the segment that overlaps most with the motion segments

        ## "explain away" by setting remaining pixels to a new value
        print("flwo", pred_motion.shape, pred_motion.dtype)
        print("segs", pred_segments.shape, pred_segments.dtype)

        return pred_motion.abs().mean()

    def forward(self, *args, **kwargs):

        pred_motion = self.model(*args, **kwargs)
        if isinstance(pred_motion, (list, tuple)):
            pred_motion = pred_motion[-1]
        pred_segments = self.segment_model(*args, **kwargs)[:,None] # [B,1,H,W]
        score = self.score_prediction(pred_motion, pred_segments)
        return score.item()

def score_functions(name):
    funcs = dict([
        ('total_energy', TotalEnergyScore),
        ('max_energy', MaximumEnergyScore),
        ('thresh_energy', ThresholdedEnergyScore),
        ('explained_motion', ExplainedAwayMotion)
    ])
    try:
        return funcs[name]
    except:
        raise ValueError("%s is not one of the valid scoring modules: %s" % (name, funcs.keys()))

def get_score_config(args):

    func = score_functions(args.score_func)
    ## todo: read in a config file
    kwargs = getattr(func, 'defaults', {})
    return (func, kwargs)

class TrainingMovieFinder(object):
    """a class for reading in movies from a Torch dataset and filtering according to motion"""
    def __init__(self,
                 model: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 outfile: str = 'training_frames.json',
                 video_length: int = 2,
                 num_files: int = None,
                 score_config=(TotalEnergyScore, {}),
                 model_call_kwargs = {'iters': 12, 'test_mode': True}
    ):
        self.model = model
        self.model_call_kwargs = model_call_kwargs
        self.dataset = dataset
        self.outfile = Path(outfile)

        self.infiles = self.dataset.files
        if num_files is not None:
            self.infiles = self.infiles[:num_files]
        self.num_files = len(self.infiles)

        self.video_length = video_length

        ## set the score fn
        self.set_score_fn(score_config)

    def set_score_fn(self, config):
        if hasattr(config, 'keys'):
            func = config.pop('func', TotalEnergyScore)
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

        print("Using Score Function --- %s" % type(self.score_fn).__name__)

        self._score_offset = getattr(self.score_fn, 'frame_offset', 0)

    def _score_video(self, video):
        n_frames = len(video)
        to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
        for i in range(n_frames - 1):
            img1, img2 = to_tensor(video[i]), to_tensor(video[i+1])
            score = self.score_fn(img1, img2, **self.model_call_kwargs)
        return score

    def filter_scores(self, scores):
        # return sorted([k for k,v in scores.items() if v > 20])
        return [int(np.argmax(np.array([scores[k] for k in sorted(scores.keys())])))]

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
        filtered = self.filter_scores(scores)
        return (filename, filtered, frame)

    def _write_scores(self, filename, frames_list):
        write_dict = {filename: frames_list}
        write_str = json.dumps(write_dict, indent=4)
        self.outfile.write_text(write_str, encoding='utf-8')

    def __call__(self, files=None, overwrite=False, verbose=False):

        if files is None:
            files = range(len(self.infiles))

        self.outdata = {}
        if not overwrite and self.outfile.exists():
            self.outdata.update(json.loads(self.outfile.read_text()))

        _fname = lambda s: '/'.join(s.split('/')[-2:])

        for idx in tqdm(files):
            if (not overwrite):
                if _fname(self.infiles[idx % self.num_files]) in self.outdata.keys():
                    continue
            filename, filtered, num_frames = self.filter_video(idx)
            filename = Path(filename)
            filename = '/'.join([filename.parent.name, filename.name])
            self.outdata[filename] = filtered
            if verbose:
                print("selected frames for file %s of length %d frames --- %s" % \
                      (filename, num_frames, filtered))


            json_str = json.dumps(self.outdata, indent=4)
            try:
                self.outfile.write_text(json_str, encoding='utf-8')
            except:
                raise Exception("unable to write to %s" % self.outfile.name)

## TODO
## read in a file
## for each movie of length (eval_len):
## read movie into memory
## pass image pairs through model to get preds
## compute metric on preds
## store data for this video
## apply threshold and/or max
## append to json outfile
def main(args):

    torch.manual_seed(1234)
    np.random.seed(1234)
    if not os.path.isdir('datasets/supervision_frames'):
        os.mkdir('datasets/supervision_frames')

    model = load_model(args)
    dataset = get_dataset(args)
    score_config = get_score_config(args)

    finder = TrainingMovieFinder(
        model=model,
        dataset=dataset,
        outfile=args.outfile,
        video_length=getattr(model, 'num_input_frames', 2),
        num_files=args.num_files,
        score_config=score_config
    )

    finder(files=None, overwrite=args.overwrite, verbose=args.verbose)
    print("Wrote training frames to %s" % args.outfile)


if __name__ == '__main__':
    args = get_args()
    # main(args)

    model = load_model(args)
    dataset = get_dataset(args)

    knet = nn.DataParallel(KMeans(32, 50, True), device_ids=args.gpus).cuda()
    seg_model = VideoSegmentationModel(model=knet, num_input_frames=1)

    ex = 3
    img1, img2, _, _ = dataset[ex]
    img1 = img1[None].cuda()
    img2 = img2[None].cuda()
    segs = seg_model(img2)

    print(segs.shape, segs.dtype)

    explain_net = ExplainedAwayMotion(model, seg_model)
    score = explain_net(img1, img2, test_mode=True, iters=12)
    print(score)

    # torch.manual_seed(1234)
    # np.random.seed(1234)

    # if not os.path.isdir('datasets/supervision_frames'):
    #     os.mkdir('datasets/supervision_frames')

    # model = load_model(args)
    # dataset = get_dataset(args)
    # print(dataset.files[:3])

    # finder = TrainingMovieFinder(model, dataset)
    # finder(None, overwrite=True)
    # for i in range(2):
    #     filename, filtered = finder.filter_video(i)
    #     print(filename, filtered)
    #     finder._write_scores(filename, filtered)
