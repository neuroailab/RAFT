import os, sys
sys.path.append('./core')

import h5py
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raft import RAFT
from bootraft import BootRaft

from argparse import ArgumentParser

def get_args(cmd=None):
    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile", type=str, help="name of outfile")
    parser.add_argument("--model", type=str, default="RAFT", help="model class")
    parser.add_argument("--checkpoint", type=str,
                        default="models/raft-sintel.pth")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--small', action='store_true', help='use small model')

    if cmd is None:
        args = parser.parse_args()
        print(args)
    else:
        args = parser.parse_args(cmd)
    return args

def load_model(args):

    if args.model.lower() == 'raft':
        model_cls = RAFT
    elif args.model.lower() == 'bootraft':
        model_cls = BootRaft
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

if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('datasets/supervision_frames'):
        os.mkdir('datasets/supervision_frames')

    model = load_model(args)
