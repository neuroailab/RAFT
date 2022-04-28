"""
Adapt from Detectron
"""
# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from detectron2.layers import CNNBlockBase, Conv2d, get_norm, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import (
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock,
    ResNet,
)
from detectron2.projects.deeplab.resnet import DeepLabStem
import torch.nn as nn
import torch


def build_resnet_deeplab_backbone():
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?


    # fmt: off
    norm                = 'BN'                      # cfg.MODEL.RESNETS.NORM
    stem_type           = 'deeplab'                     # cfg.MODEL.RESNETS.STEM_TYPE
    freeze_at           = 0                             # cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = ['res2', 'res3', 'res4']      # cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = 50                            # cfg.MODEL.RESNETS.DEPTH
    num_groups          = 1                             # cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = 64                            # cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = 128                           # cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = 256                           # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = False                         # cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res4_dilation       = 1                             # cfg.MODEL.RESNETS.RES4_DILATION
    res5_dilation       = 2                             # cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = [False] * 4                   # cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = False                         # cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = 1                             # cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    res5_multi_grid     = [1, 2, 4]                     # cfg.MODEL.RESNETS.RES5_MULTI_GRID
    # fmt: on

    input_shape = ShapeSpec(channels=3)

    if stem_type == "basic":
        stem = BasicStem(
            in_channels=input_shape.channels,
            out_channels=in_channels,
            norm=norm,
        )
    elif stem_type == "deeplab":
        stem = DeepLabStem(
            in_channels=input_shape.channels,
            out_channels=in_channels,
            norm=norm
        )
    else:
        raise ValueError("Unknown stem type: {}".format(stem_type))

    assert res4_dilation in {1, 2}, "res4_dilation cannot be {}.".format(res4_dilation)
    assert res5_dilation in {1, 2, 4}, "res5_dilation cannot be {}.".format(res5_dilation)
    if res4_dilation == 2:
        # Always dilate res5 if res4 is dilated.
        assert res5_dilation == 4

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        if stage_idx == 4:
            dilation = res4_dilation
        elif stage_idx == 5:
            dilation = res5_dilation
        else:
            dilation = 1
        first_stride = 1 if idx == 0 or dilation > 1 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        if stage_idx == 5:
            stage_kargs.pop("dilation")
            stage_kargs["dilation_per_block"] = [dilation * mg for mg in res5_multi_grid]
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)


class ResNetFPN(nn.Module):
    def __init__(self, min_level, max_level, channels={2: 256, 3: 512, 4: 1024}, latent_dim=512):
        super(ResNetFPN, self).__init__()

        self.levels = list(range(min_level, max_level + 1))
        self.min_level = min_level
        self.max_level = max_level

        # Backbone
        self.backbone = build_resnet_deeplab_backbone()

        # lateral conv for multi-level features concatenation
        for level in self.levels:
            setattr(self, 'lateral_conv_%d' % level, Conv2d(channels[level], latent_dim, kernel_size=1, bias=True))
            nn.init.kaiming_normal_(getattr(self, 'lateral_conv_%d' % level).weight, mode='fan_out', nonlinearity='relu')

        self.output_dim = latent_dim * len(self.levels)

    def forward(self, x):
        features = self.backbone(x)
        output = self.concat_multi_level(features)  # construct feature pyramid instead

        return output[self.min_level]

    def concat_multi_level(self, features):

        feats_lateral = {l: getattr(self, 'lateral_conv_%d' % l)(features['res%d' % l]) for l in self.levels}

        # Adds top-down path.
        backbone_max_level = self.levels[-1]
        feats = {backbone_max_level: feats_lateral[backbone_max_level]}

        for level in range(backbone_max_level - 1, self.min_level - 1, -1):
            feats[level] = torch.cat([
                F.interpolate(feats[level + 1], scale_factor=2.0, mode='nearest'),
                feats_lateral[level]
            ], 1)

        return feats

