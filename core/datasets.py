# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
from pathlib import Path
import numpy as np
import h5py, json
import io
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import cv2 as cv

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

## to wrap
# from dorsalventral.data.robonet import (RobonetDataset,
#                                         ROBONET_DIR,
#                                         get_robot_names)
from dorsalventral.data.davis import (DavisDataset,
                                      get_dataset_names)
from dorsalventral.data.movi import MoviDataset
from dorsalventral.data.utils import ToTensor, RgbToIntSegments

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

    IMAGE_SHAPE = (512,512,3)
    PASSES_DICT = {'images': '_img', 'flows': '_flow', 'objects': '_id', 'depths': '_depth'}

    def __init__(self,
                 root='datasets/playroom_large_v3copy/',
                 dataset_names=['model_split_4'],
                 filepattern="*",
                 test_filepattern="*9",
                 delta_time=1,
                 min_start_frame=5,
                 max_start_frame=5,
                 select_frames_per_file=True,
                 split='training',
                 training_frames=None,
                 testing_frames=None,
                 get_gt_flow=False,
                 get_gt_segments=False,
                 get_backward_frame=False,
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

        ## the frames for training given by a json file
        self.training_frames = training_frames
        self.testing_frames = testing_frames

        self.delta_time = self.dT = delta_time
        self.min_start_frame = min_start_frame
        self.max_start_frame = max_start_frame
        self.get_backward_frame = get_backward_frame
        self.select_frames_per_file = select_frames_per_file
        self.scale_to_pixels = scale_to_pixels

        if split != 'training':
            self.is_test = True

        self.get_gt_flow = get_gt_flow or (not self.is_test)
        self.get_gt_segments = get_gt_segments

    def transform_segments(self, x):
        return ToTensor()(x)
        # return transforms.Compose([
        #     ToTensor(), RgbToIntSegments()])(x)

    def __len__(self):
        return len(self.train_files if not self.is_test else self.test_files)

    @property
    def files(self):
        return self.train_files if not self.is_test else self.test_files

    @property
    def training_frames(self):
        return self._training_frames
    @training_frames.setter
    def training_frames(self, frames_file_list):
        if not isinstance(frames_file_list, (list, tuple)):
            frames_file_list = [frames_file_list]

        self._training_frames = {}
        for file in frames_file_list:
            self._training_frames.update(
                self.set_frame_selection(file, True))

    @property
    def testing_frames(self):
        return self._testing_frames
    @testing_frames.setter
    def testing_frames(self, frames_file_list):
        if not isinstance(frames_file_list, (list, tuple)):
            frames_file_list = [frames_file_list]

        self._testing_frames = {}
        for file in frames_file_list:
            self._testing_frames.update(
                self.set_frame_selection(file, False))

    def set_frame_selection(self, frames_file, training=True):

        files = self.train_files if training else self.test_files
        if (frames_file is None):
            return {fname: None for fname in files}
        elif not Path(frames_file).exists():
            raise IOError("the frames file %s does not exist" % frames_file)

        filtered_frames = json.loads(Path(frames_file).read_text())
        frames_to_use = {}

        _short = lambda s: Path(s).parent.name + '/' + Path(s).name

        ## require a string match on filename only for fname and parent dir
        for fname in files:
            frames_to_use[fname] = filtered_frames.get(_short(fname), None)

        return frames_to_use

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

    def _get_pass(self, f, pass_name, frame = 0, return_zeros=True):
        try:
            _img = f['frames'][str(frame).zfill(4)]['images'][TdwFlowDataset.PASSES_DICT.get(pass_name, pass_name)]
            _img = Image.open(io.BytesIO(_img[:]))
            _img = np.array(_img)
        except:
            if return_zeros:
                _img = np.zeros(self.IMAGE_SHAPE, dtype=np.uint8)
            else:
                return None
        return _img

    def _get_image(self, f, frame = 0, return_zeros=True):
        return self._get_pass(f, "images", frame=frame, return_zeros=return_zeros)
    def _get_image_pair(self, f, frame = 0, delta_time = 1):
        img1 = self._get_pass(f, "images", frame)
        try:
            img2 = self._get_pass(f, "images", frame + delta_time)
        except:
            img2 = img1
        return (img1, img2)

    def get_video(self, f, frame_start = 0, video_length = 2):

        video = []
        frame_end = frame_start + self.dT * video_length
        for frame in range(frame_start, frame_end, self.dT):
            img = self._get_image(f, frame, False)
            if img is None:
                return None
            video.append(img)
        return video

    def _get_flow(self, f, frame = 0):
        flow = self._get_pass(f, "flows", frame)
        flow = self.rgb_to_xy_flows(flow, to_xy=True, scale_to_pixels=self.scale_to_pixels)
        return flow.astype(np.float32)

    def _get_objects(self, f, frame = 0):
        objects = self._get_pass(f, "objects", frame)
        return self.transform_segments(objects)

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
        filtered_frames = self.testing_frames if self.is_test else self.training_frames
        index = index % len(fs)
        fname = fs[index]

        ## open the file and figure out how many frames
        f = h5py.File(fname, 'r')
        frames = sorted(list(f['frames'].keys()))
        num_frames = len(frames)

        ## choose a frame to read
        min_frame = min(self.min_start_frame or 0, num_frames - self.dT - 1)
        max_frame = min((self.max_start_frame or (num_frames - 2*self.dT)) + self.dT, num_frames - self.dT)

        frames_list = None
        if self.select_frames_per_file:
            frames_list = filtered_frames.get(fname, [])
            if frames_list is not None:
                if (len(frames_list) == 0):
                    frames_list = None

        if frames_list is not None:
            i_frame = random.choice(frames_list)
            i_frame = min(max(min_frame, i_frame), max_frame - 1)
        else:
            i_frame = np.random.randint(min_frame, max_frame)

        ## get a pair of images
        img1, img2 = self._get_image_pair(f, i_frame, self.dT)
        if self.get_backward_frame:
            img0 = self._get_image(f, i_frame - self.dT)
        else:
            img0 = None

        ## get the flow
        flow = self._get_flow(f, i_frame) if self.get_gt_flow else {}

        ## get the segments
        segments = self._get_objects(f, i_frame) if self.get_gt_segments else {}

        ## close the hdf5
        f.close()

        if self.is_test and (img0 is None):
            return (torch.from_numpy(img1).permute(2, 0, 1).float(),
                    torch.from_numpy(img2).permute(2, 0, 1).float(),
                    torch.from_numpy(flow).permute(2, 0, 1).float() if self.get_gt_flow else {},
                    segments
            )


        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        valid = (np.abs(flow[...,0]) < 1000) & (np.abs(flow[...,1]) < 1000)

        to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1).float()

        if not self.get_gt_segments and (img0 is None):
            return (to_tensor(img1), to_tensor(img2), to_tensor(flow), torch.from_numpy(valid).float())
        elif (img0 is not None):
            return (to_tensor(img1), to_tensor(img2), to_tensor(img0), torch.from_numpy(valid).float())
        elif self.get_gt_segments:
            return (to_tensor(img1), to_tensor(img2), to_tensor(flow), segments)

class TdwPngDataset(TdwFlowDataset):

    def __init__(self,
                 root='/mnt/fs6/honglinc/dataset/tdw_playroom_small/',
                 split='training',
                 dataset_names='[0-9]*',
                 test_dataset_names='[0-3]',
                 filepattern='*',
                 test_filepattern='*9',
                 delta_time=1,
                 min_start_frame=5,
                 max_start_frame=5,
                 get_gt_flow=False,
                 get_gt_segments=False,
                 get_backward_frame=False,
                 scale_to_pixels=True,
                 training_frames=None,
                 testing_frames=None,
                 aug_params=None):
        FlowDataset.__init__(self, aug_params)

        ## set files
        self.split = split
        self.training = (split == 'training')
        meta_path = os.path.join(root, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        self.train_files = sorted(
            glob(os.path.join(root, 'images',
                              'model_split_'+dataset_names,
                              filepattern)))

        self.test_files = sorted(
            glob(os.path.join(root, 'images',
                              'model_split_'+test_dataset_names,
                              test_filepattern)))

        self.is_test = (not self.training)

        # frames and which tensors to get
        self.delta_time = self.dT = delta_time
        self.delta_time = self.dT = delta_time
        self.min_start_frame = min_start_frame
        self.max_start_frame = max_start_frame
        self.get_backward_frame = get_backward_frame
        self.scale_to_pixels = scale_to_pixels

        self.get_gt_flow = get_gt_flow or (not self.is_test)
        self.get_gt_segments = get_gt_segments
        if self.get_gt_segments:
            self.transform_segments = transforms.Compose([
                ToTensor(), RgbToIntSegments()])

        ## the frames for training given by a json file
        self.training_frames = training_frames
        self.testing_frames = testing_frames

    def __len__(self):
        return len(self.file_list)

    def eval(self):
        self.is_test = True
    def train(self, do_train=True):
        self.is_test = not do_train

    def _get_pass(self, f, pass_name, frame = 0, return_zeros=True):
        pass

    def _get_image(self, f, frame = 0, return_zeros=True):
        pass

    def _get_image_pair(self, f, frame = 0, delta_time = 1):
        pass

    def _get_flow(self, f, frame = 0):
        pass

    def _get_objects(self, f, frame = 0):
        pass

    def get_video(self, f, frame_start = 0, video_length = 2):
        pass

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        if meta_key in self.meta.keys():
            zone_id = int(self.meta[meta_key]['zone'])
            raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        # gt_moving_mask
        if meta_key in self.meta.keys() and 'moving' in self.meta[meta_key].keys():
            gt_moving = raw_segment_map == int(self.meta[meta_key]['moving'])
        else:
            gt_moving = None

        return raw_segment_map, segment_map, gt_moving

    # def __getitem__(self, idx)


class TdwAffinityDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir='/mnt/fs6/honglinc/dataset/tdw_playroom_small/',
                 aff_r=5,
                 training=False,
                 phase='val',
                 size=None,
                 splits='[0-9]*',
                 test_splits='[0-3]', # trainval: 4
                 filepattern='*[0-8]',
                 test_filepattern='*9', # trainval: "0*[0-4]"
                 delta_time=1,
                 frame_idx=5,
                 raft_ckpt=None,
                 raft_args={'test_mode': True, 'iters': 24},
                 flow_thresh=0.5,
                 full_supervision=False,
                 single_supervision=False,
                 mean=None,
                 std=None,
                 is_test=True
    ):
        self.training = training
        self.phase = phase
        self.frame_idx = frame_idx
        self.delta_time = delta_time

        self.aff_r = aff_r
        self.num_levels = aff_r

        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = sorted(glob(os.path.join(dataset_dir, 'images',
                                                    'model_split_'+splits,
                                                    filepattern)))
        elif self.phase == 'val':
            self.file_list = sorted(glob(os.path.join(dataset_dir, 'images',
                                                    'model_split_'+test_splits,
                                                    test_filepattern)))
        elif self.phase == "safari":
            self.file_list = sorted(glob(os.path.join(dataset_dir, 'images', 'playroom_simple_v7safari', '*')))
        elif self.phase == "cylinder":
            self.file_list = sorted(glob(os.path.join(dataset_dir, 'images', 'cylinder_miss_contain_boxroom', '*')))


        self.precomputed_raft = False
        if not (single_supervision or full_supervision):
            self.raft = self._load_raft(raft_ckpt)
            self.raft_args = copy.deepcopy(raft_args)
        self.flow_thresh = flow_thresh

        ## normalizing
        if (mean is not None) and (std is not None):
            norm = transforms.Normalize(mean=mean, std=std)
            self.normalize = lambda x: (norm(x.float() / 255.))
        else:
            self.normalize = lambda x: x.float()

        ## resizing
        self.size = size
        if self.size is None:
            self.resize = self.resize_labels = nn.Identity()
        else:
            self.resize = transforms.Resize(self.size)
            self.resize_labels = transforms.Resize(self.size,
                                                   interpolation=transforms.InterpolationMode.NEAREST)


        ## how to get supervision inputs
        self.full_supervision = full_supervision
        self.single_supervision = single_supervision

        self.is_test = is_test

    def _load_raft(self, ckpt):
        if ckpt is None:
            self.precomputed_raft = True
            return None
        self.precomputed_raft = False
        raft = train.load_model(
            load_path=ckpt,
            small=False,
            cuda=True,
            train=False)
        return raft

    def get_raft_flow(self, img1, img2):
        assert self.raft is not None
        _, pred_flow = self.raft(img1[None].cuda().float(), img2[None].cuda().float(),
                                 **self.raft_args)
        return pred_flow

    def get_raft_mask(self, img1, img2):
        pred_flow = self.get_raft_flow(img1, img2)
        pred_mask = (pred_flow.square().sum(-3).sqrt() > self.flow_thresh)
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        return pred_mask.cpu()

    def get_semantic_map(self, motion_mask, background_class=80):
        """Only two categories: foreground (i.e. moving or moveable) and background"""
        size = motion_mask.shape[-2:]
        fg = (motion_mask.long() == 1).view(1, *size)
        return torch.cat([fg, torch.logical_not(fg)], 0).float()


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_1 = self.read_frame(file_name, frame_idx=self.frame_idx)
        try:
            image_2 = self.read_frame(file_name, frame_idx=self.frame_idx+self.delta_time)
        except Exception as e:
            image_2 = image_1.clone()  #  This will always give empty motion segments, so the loss will be zero
            print('Encounter error:', e)
        segment_colors = self.read_frame(file_name.replace('/images/', '/objects/'), frame_idx=self.frame_idx)
        _, segment_map, gt_moving = self.process_segmentation_color(segment_colors, file_name)

        if self.full_supervision:
            moving = (segment_map > 0)
        elif self.precomputed_raft:
            pred_flow = self.read_flow(file_name, size=self.size)
            moving = (pred_flow.square().sum(-3).sqrt() > self.flow_thresh)
        elif self.raft is not None:
            moving = self.get_raft_mask(image_1, image_2)
        else:
            assert self.single_supervision
            # raise NotImplementedError("Don't use the GT moving object!")
            moving = gt_moving

        if self.is_test:
            return (self.normalize(self.resize(image_1).float()), segment_map, gt_moving)

        semantic = self.get_semantic_map(moving)
        aff_target = self.get_affinity_target(segment_map if self.full_supervision else moving)

        return (self.normalize(self.resize(image_1).float()), self.resize_labels(semantic), aff_target)

    def get_affinity_target(self, segments):
        assert len(segments.shape) == 2, segments.shape
        segments = segments.long()
        segments = self.resize_labels(segments[None])[0] # [H,W] <long>

        aff_targets = torch.zeros((self.num_levels, self.aff_r**2, self.size[0], self.size[1])).float()
        for lev in range(self.num_levels):
            segs_lev = segments[0:self.size[0]:2**lev,
                                0:self.size[1]:2**lev]
            size = [self.size[0] // (2**lev), self.size[1] // (2**lev)]

            segs_aff_lev_2_pix = torch.zeros((size[0] + (self.aff_r//2)*2,
                                              size[1] + (self.aff_r//2)*2)).long()
            segs_aff_lev_2_pix[self.aff_r//2:
                               size[0]+self.aff_r//2,
                               self.aff_r//2:
                               size[1]+self.aff_r//2] = segs_lev

            segs_aff_compare = torch.zeros((self.aff_r**2, size[0], size[1])).long()

            ## set affinity values
            for i in range(self.aff_r):
                for j in range(self.aff_r):
                    segs_aff_compare[i*self.aff_r + j] = segs_aff_lev_2_pix[i:i+size[0],
                                                                            j:j+size[1]]

            ## compare
            aff_t = (segs_lev[None] == segs_aff_compare).float()
            aff_targets[lev, :, 0:size[0], 0:size[1]] = aff_t

        return aff_targets.to(segments.device)

    @staticmethod
    def read_frame(path, frame_idx):
        image_path = os.path.join(path, format(frame_idx, '05d') + '.png')
        return read_image(image_path)

    @staticmethod
    def read_flow(path, size=[256,256]):
        flow_path = path.replace('/images/', '/flows/') + '.pt'
        try:
            raft_flow = torch.load(flow_path)
        except:
            raft_flow = torch.zeros((2, *size))
        return raft_flow.float()

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        if meta_key in self.meta.keys():
            zone_id = int(self.meta[meta_key]['zone'])
            raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        # gt_moving_mask
        if meta_key in self.meta.keys() and 'moving' in self.meta[meta_key].keys():
            gt_moving = raw_segment_map == int(self.meta[meta_key]['moving'])
        else:
            gt_moving = None

        return raw_segment_map, segment_map, gt_moving

class MoviFlowDataset(MoviDataset):

    def __init__(self,
                 root,
                 split='train',
                 sequence_length=3,
                 passes=["images", "objects", "flow"],
                 *args, **kwargs):
        super().__init__(dataset_dir=root,
                         split=split,
                         sequence_length=sequence_length,
                         passes=passes,
                         *args, **kwargs)

        self.is_test = ('train' not in self.split)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        video = data['images'].float()
        img1, img2 = video[-2], video[-1] # index and forward frames
        img0 = None
        if video.shape[0] == 3:
            img0 = video[0]
        elif self.is_test and (img0 is None):
            minv, maxv = self.meta['forward_flow_range']
            img0 = data['flow'][-2] / 65535 * (maxv - minv) + minv

        segments = data.get('objects', None)
        if segments is not None:
            segments = segments[-2] # index frame

        return (img1, img2, img0, segments)

# class RobonetFlowDataset(RobonetDataset):

#     all_robots = get_robot_names()
#     def __init__(self,
#                  root=ROBONET_DIR,
#                  dataset_names=all_robots,
#                  sequence_length=2,
#                  *args, **kwargs):
#         if dataset_names is None:
#             dataset_names = self.all_robots
#         super().__init__(dataset_dir=root,
#                          dataset_names=dataset_names,
#                          sequence_length=sequence_length,
#                          *args, **kwargs)


#     def __getitem__(self, idx):

#         data_dict = super().__getitem__(idx)
#         img1, img2 = data_dict['images'][:2].split([1,1], 0)
#         return img1[0].float(), img2[0].float()

#     def get_video(self, f, frame_start = 0, num_frames = 2):
#         meta = self.meta_data_frame[self.meta_data_frame.index == Path(str(f.filename)).name]
#         video_length = int(meta['img_T'])
#         if (frame_start + num_frames) > video_length:
#             return None
#         video = self.get_movie(f, meta, frame=frame_start, num_frames=num_frames, transform={})
#         return list(video)

# class DavisFlowDataset(DavisDataset):

#     all_dataset_names = get_dataset_names()
#     def __init__(self,
#                  root='/data5/dbear/DAVIS2016',
#                  dataset_names=all_dataset_names,
#                  sequence_length=2,
#                  get_gt_flow=False,
#                  flow_gap=1,
#                  *args, **kwargs):
#         if dataset_names is None:
#             dataset_names = self.all_dataset_names
#         super().__init__(dataset_dir=root,
#                          dataset_names=dataset_names,
#                          sequence_length=sequence_length,
#                          to_tensor=True,
#                          get_flows=get_gt_flow,
#                          flow_gap=flow_gap,
#                          *args, **kwargs)

#         ## center crop
#         size = self.resize_to or (1080, 1920)
#         crop_size = (8 * (size[0] // 8), 8 * (size[1] //  8))
#         if crop_size == size:
#             self.crop = nn.Identity()
#         else:
#             self.crop = transforms.CenterCrop(crop_size)

#     def __getitem__(self, idx):
#         data_dict = super().__getitem__(idx)
#         if 'segments' in data_dict.keys():
#             self.gt = self.crop(data_dict['segments'])
#         if 'flow_images' in data_dict.keys():
#             self.flow_images = self.crop(data_dict['flow_images'])

#         img1, img2 = [self.crop(im) for im in list(data_dict['images'][:2])]
#         if self.get_flows:
#             flow = self.crop(data_dict['flows'][0])
#             return img1.float(), img2.float(), flow
#         else:
#             return img1.float(), img2.float()

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'tdw':
        if args.no_aug:
            aug_params = None
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}

        if args.full_playroom:
            root = 'datasets/playroom_large_v3full/'
            dataset_names = ['model_split_%d' % d for d in range(32)]
        else:
            root = 'datasets/playroom_large_v3copy/'
            dataset_names = ['model_split_4']

        train_dataset = TdwFlowDataset(
            aug_params=aug_params,
            split='training',
            root=root,
            dataset_names=dataset_names,
            filepattern=args.filepattern or "*",
            test_filepattern=args.test_filepattern or "*9",
            min_start_frame=6 if (args.model in ['motion', 'occlusion', 'thingness', 'boundary', 'flow']) else 5,
            max_start_frame=(args.max_frame if args.max_frame > 0 else None),
            training_frames=args.training_frames,
            get_backward_frame=((args.model in ['motion','occlusion', 'thingness', 'boundary', 'flow']) and not args.supervised)
        )

    elif 'movi' in args.stage:
        root = os.path.join(
            args.dataset_dir,
            args.stage,
            '256x256' if args.image_size[0] > 128 else '128x128',
            '1.0.0'
        )
        train_dataset = MoviFlowDataset(
            root=root,
            split=args.split,
            sequence_length=3,
            delta_time=args.flow_gap,
            passes=['images', 'objects'],
            min_start_frame=0,
            max_start_frame=None
        )

        print("training on %s with %d movies" % (args.stage, len(train_dataset.ds)))

    # elif args.stage == 'robonet':
    #     print("dataset names", args.dataset_names)
    #     train_dataset = RobonetFlowDataset(
    #         root=ROBONET_DIR,
    #         dataset_names=args.dataset_names,
    #         sequence_length=2,
    #         min_start_frame=0,
    #         imsize=None,
    #         train=True,
    #         filter_imsize=args.image_size
    #     )

    # elif args.stage == 'davis':
    #     print("dataset names", args.dataset_names)
    #     train_dataset = DavisFlowDataset(
    #         root='/data5/dbear/DAVIS2016',
    #         dataset_names=None,
    #         sequence_length=2,
    #         split=args.train_split,
    #         resize=args.image_size,
    #         get_gt_flow=True,
    #         flow_gap=args.flow_gap)

    elif args.stage == 'chairs':
        if args.no_aug:
            aug_params = None
        else:
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

    if 'movi' not in args.stage:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
            pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    elif 'movi' in args.stage:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    num_examples = len(train_dataset)

    print('Training with %d image pairs' % num_examples)
    return (train_loader, num_examples)

if __name__ == '__main__':
    root = os.path.join(
        '/mnt/fs6/honglinc/dataset/tensorflow_datasets/',
        'movi_e',
        '256x256',
        '1.0.0'
    )
    train_dataset = MoviFlowDataset(
        root=root,
        split='test',
        sequence_length=3,
        delta_time=1,
        passes=['images', 'objects', 'flow'],
        min_start_frame=0,
        max_start_frame=24
    )
    train_loader = iter(data.DataLoader(train_dataset, batch_size=1, shuffle=False))
    for i in range(5):
        data = train_loader.next()
        for v in data:
            if v is not None:
                print(v.dtype, v.shape, v.amin().item(), v.amax().item())
