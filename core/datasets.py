# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import numpy as np
import h5py
import io
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2 as cv

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

import kornia.color

def rgb_to_xy_flows(flows, to_image_coordinates=True, to_sampling_grid=False):
    assert flows.dtype == torch.uint8, flows.dtype
    assert flows.shape[-3] == 3, flows.shape
    flows_hsv = kornia.color.rgb_to_hsv(flows.float() / 255.)

    hue, sat, val = flows_hsv.split([1, 1, 1], dim=-3)
    flow_x = torch.cos(hue) * val
    flow_y = torch.sin(hue) * val

    if to_image_coordinates:
        flow_h = -flow_y
        flow_w = flow_x
        return torch.cat([flow_h, flow_w], -3)
    elif to_sampling_grid:
        return torch.cat([flow_x, -flow_y], -3)
    else:
        return torch.cat([flow_x, flow_y], -3)

class RgbFlowToXY(object):
    def __init__(self, to_image_coordinates=True, to_sampling_grid=False):
        self.to_image_coordinates = to_image_coordinates
        self.to_sampling_grid = to_sampling_grid
    def __call__(self, flows_rgb):
        return rgb_to_xy_flows(flows_rgb, self.to_image_coordinates, self.to_sampling_grid)

class FlowToRgb(object):

    def __init__(self, max_speed=1.0, from_image_cooordinates=True, from_sampling_grid=False):
        self.max_speed = max_speed
        self.from_image_cooordinates = from_image_cooordinates
        self.from_sampling_grid = from_sampling_grid

    def __call__(self, flow):
        assert flow.size(-3) == 2, flow.shape
        if self.from_sampling_grid:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
            flow_y = -flow_y
        elif not self.from_image_cooordinates:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
        else:
            flow_h, flow_w = torch.split(flow, [1,1], dim=-3)
            flow_x, flow_y = [flow_w, -flow_h]

        angle = torch.atan2(flow_y, flow_x) # in radians from -pi to pi
        speed = torch.sqrt(flow_x**2 + flow_y**2) / self.max_speed

        hue = torch.fmod(angle, torch.tensor(2 * np.pi))
        sat = torch.ones_like(hue)
        val = speed

        hsv = torch.cat([hue, sat, val], -3)
        rgb = kornia.color.hsv_to_rgb(hsv)
        return rgb


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None,
                 split='train', split_file='chairs_split.txt',
                 root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt(split_file, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

class TdwFlowDataset(FlowDataset):

    PASSES_DICT = {'images': '_img', 'flows': '_flow', 'objects': '_id', 'depths': '_depth'}

    def __init__(self,
                 root='datasets/playroom_large_v3copy/',
                 dataset_names=['model_split_4'],
                 filepattern="*",
                 test_filepattern="*9",
                 delta_time=1,
                 min_start_frame=5,
                 max_start_frame=5,
                 split='training',
                 scale_to_pixels=True,
                 aug_params=None):
        super(TdwFlowDataset, self).__init__(aug_params)

        ## set filenames
        self.datasets = [osp.join(root, nm) for nm in dataset_names]
        self.train_files, self.test_files = [], []
        for nm in self.datasets:
            self.train_files.extend(sorted(glob(osp.join(nm, filepattern + ".hdf5"))))
        for nm in self.datasets:
            self.test_files.extend(sorted(glob(osp.join(nm, test_filepattern + ".hdf5"))))

        self.delta_time = self.dT = delta_time
        self.min_start_frame = min_start_frame
        self.max_start_frame = max_start_frame
        self.scale_to_pixels = scale_to_pixels

        if split != 'training':
            self.is_test = True

    def __len__(self):
        return len(self.train_files if not self.is_test else self.test_files)

    def eval(self):
        self.is_test = True
    def train(self, do_train=True):
        self.is_test = not do_train

    @staticmethod
    def rgb_to_xy_flows(flows, to_xy=True, scale_to_pixels=False):
        assert flows.dtype == np.uint8, flows.dtype
        assert flows.shape[-1] == 3, flows.shape

        flows = cv.cvtColor(flows, cv.COLOR_RGB2HSV)
        hue, sat, val = np.split(flows, 3, axis=-1)
        ## opencv puts hue in range [0, 180] for uint8
        hue = (hue / 180.0) * 2 * np.pi
        sat = sat / 255.
        val = val / 255.

        flow_x = np.cos(hue) * val
        flow_y = np.sin(hue) * val

        if scale_to_pixels:
            H,W = flows.shape[:2]
            flow_x *= 0.5 * (W - 1)
            flow_y *= 0.5 * (H - 1)

        if to_xy:
            return np.concatenate([flow_x, flow_y], -1)
        else:
            return np.concatenate([-flow_y, flow_x], -1)

    def _get_pass(self, f, pass_name, frame = 0):
        _img = f['frames'][str(frame).zfill(4)]['images'][TdwFlowDataset.PASSES_DICT.get(pass_name, pass_name)]
        _img = Image.open(io.BytesIO(_img[:]))
        _img = np.array(_img)
        return _img
    def _get_image(self, f, frame = 0):
        return self._get_pass(f, "images", frame=frame)
    def _get_image_pair(self, f, frame = 0):
        return (self._get_pass(f, "images", frame), self._get_pass(f, "images", frame + self.dT))
    def _get_flow(self, f, frame = 0):
        flow = self._get_pass(f, "flows", frame)
        flow = self.rgb_to_xy_flows(flow, to_xy=True, scale_to_pixels=self.scale_to_pixels)
        return flow.astype(np.float32)

    def _init_seed(self):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True


    def __getitem__(self, index):

        self._init_seed()
        fs = self.test_files if self.is_test else self.train_files
        index = index % len(fs)
        fname = fs[index]

        ## open the file and figure out how many frames
        f = h5py.File(fname, 'r')
        frames = sorted(list(f['frames'].keys()))
        num_frames = len(frames)

        ## choose a frame to read
        min_frame = min(self.min_start_frame, num_frames - self.dT - 1)
        max_frame = min(self.max_start_frame + self.dT, num_frames - self.dT)
        i_frame = np.random.randint(min_frame, max_frame)

        ## get a pair of images
        img1, img2 = self._get_image_pair(f, i_frame)

        if self.is_test:
            return (torch.from_numpy(img1).permute(2, 0, 1).float(),
                    torch.from_numpy(img2).permute(2, 0, 1).float(),
                    {})


        ## get the flow
        flow = self._get_flow(f, i_frame)

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        valid = (np.abs(flow[...,0]) < 1000) & (np.abs(flow[...,1]) < 1000)

        to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1).float()

        return (to_tensor(img1), to_tensor(img2), to_tensor(flow), torch.from_numpy(valid).float())

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'tdw':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        # aug_params = None
        train_dataset = TdwFlowDataset(aug_params=aug_params, split='training')

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    print(train_dataset)
    print(len(train_dataset))
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
