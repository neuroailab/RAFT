import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from update import BasicUpdateBlock, SmallUpdateBlock, FlowHead
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

## fire propagation
import dorsalventral.models.fire_propagation as prop
import dorsalventral.models.layers as layers
import dorsalventral.models.layer_utils as layer_utils

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args, **kwargs):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout, gate_stride=args.gate_stride)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout, gate_stride=args.gate_stride)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, gate_stride=args.gate_stride)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout, gate_stride=args.gate_stride)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.classifier_head = nn.Sequential()
        self.gate_stride = self.args.gate_stride
        self.ds = 4 * self.args.gate_stride

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//self.ds, W//self.ds, device=img.device)
        coords1 = coords_grid(N, H//self.ds, W//self.ds, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """

        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.ds, self.ds, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.ds * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, self.ds*H, self.ds*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False, output_hidden=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            if self.args.static_coords:
                corr = corr_fn(coords0.detach())
            else:
                corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if output_hidden:
                flow_up = self.upsample_flow(self.classifier_head(net), up_mask)
            elif up_mask is None:
                flow_up = upflow8(self.classifier_head(coords1 - coords0))
            else:
                flow_up = self.upsample_flow(self.classifier_head(coords1 - coords0), up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return up_mask, flow_up

        return flow_predictions

class ThingsClassifier(RAFT):

    def __init__(self, args):
        super().__init__(args)
        self.classifier_head = FlowHead(input_dim=self.hidden_dim, out_dim=1)


    def forward(self, *args, **kwargs):
        ## static model
        img1, img2 = args[:2]
        return super().forward(img1, img1, *args[2:], **kwargs, output_hidden=True)

class BoundaryClassifier(RAFT):

    def __init__(self, args):
        super().__init__(args)
        out_type = self.args.orientation_type
        if out_type == 'regression':
            out_dim = 3
        elif out_type == 'classification':
            out_dim = 9
        elif out_type == 'combined':
            out_dim = 4
        elif hasattr(self.args, 'out_channels'):
            out_dim = self.args.out_channels            
        else:
            out_dim = 1

        self.classifier_head = FlowHead(
            input_dim=self.hidden_dim, out_dim=out_dim)
        self.static_input = self.args.static_input
        self.out_channels = out_dim

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            img1, img2 = args[0], args[0]
        else:
            img1, img2 = args[:2]
        out = super().forward(
            img1,
            img1 if self.static_input else img2,
            *args[2:], **kwargs, output_hidden=True
        )
        return out

class GateClassifier(BoundaryClassifier):

    def __init__(self, args):
        args.orientation_type = None
        assert hasattr(args, 'out_channels'), args
        args.static_input = True
        super(GateClassifier, self).__init__(args)

        ## final classifier
        self.gate_head = nn.Conv2d(self.out_channels, 1, kernel_size=1)

    def forward(self, *args, get_features=False, **kwargs):

        hidden = super().forward(*args, **kwargs)
        if kwargs.get('test_mode', False):
            up_mask, features = hidden
            if get_features:
                return (up_mask, features)
            return (up_mask, self.gate_head(features))
        else:
            features_list = hidden
            if get_features:
                return features_list
            return list(map(self.gate_head, features_list))

class ConvClassifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            layers.ResNetBasicBlock(
                3, self.args.out_channels, stride=1, kernel_size=7, norm_fn='batch'),
            layers.ResNetBasicBlock(
                self.args.out_channels, self.args.out_channels, stride=1, kernel_size=3, norm_fn='batch')
        )
        self.decoder = nn.Conv2d(self.args.out_channels, 1 , kernel_size=1)

    def forward(self, *args, get_features=False, test_mode=False, **kwargs):

        x = args[0]
        z = self.encoder(x)
        if test_mode:
            return (None, z if get_features else self.decoder(z))
        return [z if get_features else self.decoder(z)]

class CentroidRegressor(RAFT):

    def __init__(self, args):
        super().__init__(args)
        self.classifier_head = FlowHead(
            input_dim=self.hidden_dim,
            out_dim=(3 if args.predict_mask else 2))

    def forward(self, *args, **kwargs):
        ## static model
        img1, img2 = args[:2]
        return super().forward(img1, img1, *args[2:], **kwargs, output_hidden=True)

class MotionClassifier(RAFT):
    """Like a Things Classifier but uses motion"""
    def __init__(self, args):
        super().__init__(args)
        self.classifier_head = FlowHead(input_dim=self.hidden_dim, out_dim=1)

    def set_mode(self, mode):
        self.mode = mode

    def compute_loss(preds, target, gamma=0.8, size=None):
        n_preds = len(preds)
        loss = 0.0
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        for i in range(n_preds):
            i_weight = gamma ** (n_preds - i - 1)
            i_loss = loss_fn(preds[i], target).mean()
            loss += i_weight * i_loss
        return loss

    def forward(self, *args, **kwargs):
        img1, img2 = args[:2]
        return super().forward(img1, img2, *args[2:], **kwargs, output_hidden=True)

class SpatialAffinityDecoder(RAFT):
    """A model for predicting static local affinities with a RAFT-like decoder"""
    def __init__(self, args):
        super().__init__(args)
        self.affinity_radius = self.args.affinity_radius
        self.kernel_size = self.affinity_radius*2 + 1
        self.out_dim = self.kernel_size ** 2
        self.nonlinearity = layer_utils.activation_func(self.args.affinity_nonlinearity)
        self.static_input = self.args.static_input

        self.classifier_head = FlowHead(input_dim=self.hidden_dim, out_dim=self.out_dim)

    def forward(self, *args, **kwargs):
        img1, img2 = args[:2]
        return super().forward(
            img1,
            img1 if self.static_input else img2,
            *args[2:], **kwargs, output_hidden=True)
        
    
        

