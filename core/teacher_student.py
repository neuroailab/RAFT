from functools import partial
from pathlib import Path
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from core.teachers import MotionToStaticTeacher
from core.eisen import EISEN
from teachers import get_args
import matplotlib.pyplot as plt

def _get_model_class(name):
    cls = None
    if 'motion_to_static' in name:
        cls = MotionToStaticTeacher
    elif 'eisen' in name:
        cls = EISEN
    else:
        raise ValueError("Couldn't identify a model class associated with %s" % name)
    return cls


def load_model(model_class, load_path, freeze, **kwargs):
    if model_class == 'raft_pretrained':
        return None # load from saved flows from pretrained models
    cls = _get_model_class(model_class)
    assert cls is not None, "Wasn't able to infer model class"

    # build model
    model = cls(**kwargs)

    # load checkpoint
    if load_path is not None:
        weight_dict = torch.load(load_path)
        new_dict = dict()
        for k in weight_dict.keys():
            if 'module' in k:
                new_dict[k.split('module.')[-1]] = weight_dict[k]
        did_load = model.load_state_dict(new_dict, strict=False)
        print(did_load, type(model).__name__, load_path)

    # freeze params if needed
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


class TeacherStudent(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.teacher_class in ['raft_pretrained', 'motion_to_static'], f"Unexpected teacher class {args.teacher_class}"
        print('Teacher class: ', args.teacher_class)
        self.teacher = load_model(args.teacher_class, args.teacher_load_path, freeze=True, **args.teacher_params)
        self.student = load_model(args.student_class, args.student_load_path, freeze=False, **args.student_params)
        self.args = args

    def forward(self, img1, img2, iters, raft_moving=None):

        if self.args.teacher_class == 'raft_pretrained':
            print('Using RAFT flow')
            target = raft_moving
        else:
            self.teacher.eval()
            with torch.no_grad():
                target = self.teacher(img1, img2) + 1 # add 1 so that the background has value zero

        # self.visualize_targets(img1, target)
        affinity, loss, segments = self.student(img1, target, get_segments=False)
        metric = {'loss': loss.detach()}
        return loss, metric

    @staticmethod
    def visualize_targets(img, seg):
        plt.subplot(1, 2, 1)
        plt.imshow(img[0].permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(seg[0].cpu())
        plt.axis('off')
        plt.show()
        plt.close()



if __name__ == '__main__':
    args = get_args()

    motion_params = {
        'small': False,

    }
    boundary_params = {
        'small': False,
        'static_input': False,
        'orientation_type': 'regression'
    }

    teacher_class = 'motion_to_static'
    teacher_params = {
        'downsample_factor': 4,
        'motion_path': args.motion_path,
        'motion_model_params': motion_params,
        'boundary_path': args.boundary_path,
        'boundary_model_params': boundary_params
    }

    student_class = 'eisen'
    student_params = {}


    teacher_student = TeacherStudent(
        teacher_class=teacher_class,
        teacher_params=teacher_params,
        teacher_load_path=None,
        student_class=student_class,
        student_params=student_params,
        student_load_path=None,
    )

