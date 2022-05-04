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
from core.utils.segmentation_metrics import measure_static_segmentation_metric
from teachers import get_args
import matplotlib.pyplot as plt
import os

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

    def forward(self, img1, img2, gt_segment, iters, raft_moving=None, get_segments=False):

        if self.args.teacher_class == 'raft_pretrained':
            target = raft_moving
        else:
            self.teacher.eval()
            with torch.no_grad():
                target = self.teacher(img1, img2) + 1 # add 1 so that the background has value zero

        # self.visualize_targets(img1, target)
        affinity, loss, pred_segment = self.student(img1, target, get_segments=get_segments)

        metric, visuals = self.measure_segments(pred_segment, gt_segment)
        # self.visualize_segments(visuals, target, img1)
        metric = {'miou': metric['metric_pred_segment_mean_ious']}
        return loss, metric

    def measure_segments(self, pred_segment, gt_segment):

       return measure_static_segmentation_metric({'pred_segment': pred_segment}, {'gt_segment': gt_segment}, [128, 128],
                                           segment_key=['pred_segment'],
                                           moving_only=False,
                                           eval_full_res=True)



    def visualize_segments(self, visuals, target, image, prefix=''):

        matched_cc_preds, matched_gts, cc_ious = visuals['pred_segment']

        H = W = 128

        fsz = 19
        num_plots = 2+len(matched_cc_preds[0])*2
        fig = plt.figure(figsize=(num_plots * 4, 5))
        gs = fig.add_gridspec(1, num_plots)
        ax1 = fig.add_subplot(gs[0])

        plt.imshow(image[0].permute([1, 2, 0]).cpu())
        plt.axis('off')
        ax1.set_title('Image', fontsize=fsz)


        # labels = F.interpolate(batched_inputs[0]['gt_moving'].unsqueeze(0).float().cuda(), size=[H, W], mode='nearest')
        ax = fig.add_subplot(gs[1])

        if target is None:
            target = torch.zeros(1, 1, H, W)
        plt.imshow(target[0].cpu())
        plt.title('Supervision', fontsize=fsz)
        plt.axis('off')

        for i, (cc_pred, gt, cc_iou) in enumerate(zip(matched_cc_preds[0], matched_gts[0], cc_ious[0])):
            ax = fig.add_subplot(gs[2 + i])
            ax.imshow(cc_pred)
            ax.set_title('Pred (IoU: %.2f)' % cc_iou, fontsize=fsz)
            plt.axis('off')

            ax = fig.add_subplot(gs[2 + len(matched_cc_preds[0]) + i])
            ax.imshow(gt)
            plt.axis('off')
            ax.set_title('GT %d' % i, fontsize=fsz)

        # file_idx = batched_inputs[0]['file_name'].split('/')[-1].split('.hdf5')[0]
        # save_path = os.path.join(self.vis_saved_path, 'step_%smask_%s_%s.png' % (prefix, 'eval' if iter is None else str(iter), file_idx))
        # print('Save fig to ', save_path)
        # plt.savefig(save_path, bbox_inches='tight')

        plt.show()
        plt.close()


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

