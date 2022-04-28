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
    def __init__(self,
                 teacher_class, teacher_params, teacher_load_path,
                 student_class, student_params, student_load_path):
        super().__init__()

        self.teacher_model = load_model(teacher_class, teacher_load_path, freeze=True, **teacher_params)
        self.student_model = load_model(student_class, student_load_path, freeze=False, **student_params)
        self.teacher_model.eval()


    def forward(self, x):
        img1, img2 = x[:, -2], x[:, -1]
        target = self.teacher(img1.cuda(), img2.cuda())
        affinity, loss, segments = self.student_model(img1, target, get_segments=False)

        return loss



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

