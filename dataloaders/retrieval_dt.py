from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import json
import hashlib
from os import path as osp
import pickle

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize

from registry import DATASETS


@DATASETS.register_module()
class RetrievalDataset(Dataset):
    def __init__(
        self, ann, data_root='data', resolution=224, 
        split='train', max_frames=12, slice_framepos=2, 
    ):
        self.ann = json.load(open(osp.join(data_root, ann)))
        self.frames = {}
        self.data_root = data_root
        self.resolution = resolution
        self.split = split
        self.max_frames = max_frames
        self.slice_framepos = slice_framepos
        self.transform = Compose([
            Resize(resolution, interpolation=Image.BICUBIC),
            CenterCrop(resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index) -> dict:
        item_info = self.ann[index]    # {'caption': , 'id': , 'vid': , 'frames': }
        video_name = item_info['vid']
        frame_paths = [osp.join(self.data_root, f) for f in item_info['frames']]
        caption = item_info['caption']

        # sample
        frame_paths.sort()
        sample_num, sample_method = self.max_frames, self.slice_framepos
        if len(frame_paths) > sample_num:
            if sample_method == 2: 
                index = [0]
                for i in range(sample_num):
                    cur_idx = i * len(frame_paths) // sample_num
                    if not index[-1] == cur_idx: index.append(cur_idx)
                frame_paths = [frame_paths[idx] for idx in index]
            elif sample_method == 0: frame_paths = frame_paths[:sample_num]
            elif sample_method == 1: frame_paths = frame_paths[-sample_num:]
            else: raise RuntimeError('wrong sample method')
        frame_paths.sort()

        frame_pils = [Image.open(fp) for fp in frame_paths]
        img_tensors = torch.stack([self.transform(fp) for fp in frame_pils])
        mask = [1] * img_tensors.shape[0]
        if sample_num > img_tensors.shape[0]:
            mask += [0] * (sample_num - img_tensors.shape[0])
            img_tensors = torch.cat([
                img_tensors,
                torch.zeros(sample_num - img_tensors.shape[0], *img_tensors.shape[1:]),
            ])

        return dict(
            list_plus=dict(
                video_batch=img_tensors, 
                caption_batch=caption, 
                image_paths=frame_paths),
            tensor=dict(v_mask=mask),
        )
