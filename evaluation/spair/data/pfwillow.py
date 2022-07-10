r"""PF-WILLOW dataset"""
import os

import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset


class PFWillowDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split):
        r"""PF-WILLOW dataset constructor"""
        super(PFWillowDataset, self).__init__(benchmark, datapath, thres, device, split)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.src_kps = self.train_data.iloc[:, 2:22].values
        self.trg_kps = self.train_data.iloc[:, 22:].values
        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        self.cls_ids = list(map(lambda names: self.cls.index(names.split('/')[1]), self.src_imnames))
        self.src_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.trg_imnames))

    def __getitem__(self, idx):
        r"""Construct and return a batch for PF-WILLOW dataset"""
        sample = super(PFWillowDataset, self).__getitem__(idx)
        sample['pckthres'] = self.get_pckthres(sample).to(self.device)

        return sample

    def get_image(self, img_names, idx):
        r"""Return image tensor"""
        return super(PFWillowDataset, self).get_image(img_names, idx)

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        if self.thres == 'bbox':
            return max(sample['trg_kps'].max(1)[0] - sample['trg_kps'].min(1)[0]).clone()
        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size()[1], sample['trg_img'].size()[2]))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts, idx):
        r"""Return key-points of an image"""
        point_coords = pts[idx, :].reshape(2, 10)
        point_coords = torch.tensor(point_coords.astype(np.float32))

        return point_coords
