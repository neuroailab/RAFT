import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raft import RAFT
import dorsalventral.models.bootnet as bootnet
import dorsalventral.models.layers as layers
import dorsalventral.models.targets as targets

if __name__ == '__main__':

    import os

    net = nn.Sequential(
        layers.RaftVideoEncoder(in_dim=3),
        layers.RaftVideoDecoder(output_video=False)
    )

    print(layers.num_parameters(net))
