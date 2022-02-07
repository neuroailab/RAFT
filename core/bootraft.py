import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raft import RAFT
import dorsalventral.models.bootnet as bootnet
import dorsalventral.models.layers as layers
import dorsalventral.models.targets as targets

class BootRaft(nn.Module):
    """Wraps bootnet implementatino of RAFT"""
    def __init__(self, args, in_dim=3):
        super(BootRaft, self).__init__()
        self.args = args
        self.encoder = layers.RaftVideoEncoder(in_dim=in_dim)
        self.decoder = layers.RaftVideoDecoder(output_video=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):

        x = torch.stack([image1, image2], 1) # [B,2,3,H,W]
        x = self.encoder(x)

        ## set num iters
        self.decoder.num_iters = iters
        self.decoder.output_predictions = not test_mode
        self.decoder.train(not test_mode)
        x = self.decoder(x)

        if test_mode:
            return (self.decoder.delta_coords, x)
        return x


if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'

    net = BootRaft(None).cuda()


    # net = nn.Sequential(
    #     layers.RaftVideoEncoder(in_dim=3),
    #     layers.RaftVideoDecoder(output_video=False)
    # )

    print(layers.num_parameters(net))
    # print("fnet+cnet", layers.num_parameters(net[0]))
    # print("decoder", layers.num_parameters(net[1]))
    # print(net[1])

    x = torch.rand(2,2,3,512,512).cuda()
    y_list = net(x[:,0], x[:,1], iters=3, test_mode=True)
    print(len(y_list))
    print(y_list[0].shape, y_list[1].shape)
