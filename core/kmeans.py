import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import coords_grid

class KMeans(nn.Module):
    num_input_frames = 1
    def __init__(self, num_clusters, num_iters=50, append_coordinates=True):
        super().__init__()
        self.num_clusters = self.K = num_clusters
        self.num_iters = num_iters
        self.append_coordinates = append_coordinates

    def preprocess(self, x):

        shape = x.shape
        H,W = shape[-2:]
        self.B = shape[0]
        self.size = [H,W]

        # first frame only
        if len(shape) == 5:
            assert shape[1] == 1, shape
            x = x.view(shape[0], *shape[2:])
        else:
            assert len(shape) == 4, shape

        if self.append_coordinates:
            coords = coords_grid(shape[0], H, W, x.device)
            xgrid, ygrid = coords.split([1, 1], 1)
            xgrid = (2*xgrid / (W - 1)) - 1
            ygrid = (2*ygrid / (H - 1)) - 1
            coords = torch.cat([ygrid, xgrid], 1)
            x = torch.cat([x, coords], -3)

        return x

    def _initialize_cluster_centers(self, x):
        k_inds = torch.randint(low=0, high=self.N, size=[self.B,self.K], dtype=torch.long)
        b_inds = torch.arange(self.B, dtype=torch.long).unsqueeze(1).repeat(1, self.K)
        inds = torch.stack([b_inds, k_inds], 0)
        return x[list(inds)] # [B,K,D]

    def compute_labels(self, x, c):
        """
        x: data [B,N,D]
        c: centroids [B,K,D]
        """
        ## distance between points n and centroids k
        dists = (x[...,None,:] - c[:,None]).square().sum(-1) # [B,N,K]
        labels = dists.argmin(-1).long() # [B,N] in [0,K)
        return labels

    def forward(self, x):
        x = x.float() / 255.0
        x = 2.*x - 1.

        x = self.preprocess(x)
        x = x.view(self.B, -1, self.size[0] * self.size[1]).transpose(1, 2)
        self.N, self.D = x.shape[-2:]

        c = self._initialize_cluster_centers(x)
        labels = self.compute_labels(x, c)

        for it in range(self.num_iters):
            labels = F.one_hot(labels, num_classes=self.K).float() # [B,N,K]
            num_pts = labels.sum(1, True).transpose(1,2).clamp(min=1.) # [B,K,1]
            c = torch.einsum('bnd,bnk->bkd', x, labels) / num_pts
            labels = self.compute_labels(x, c)

        return labels.reshape(self.B, *self.size)
