import json
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image


class StanfordDogsDataset(Dataset):
    def __init__(self, datapath, thres, device, split='random'):
        
        super(StanfordDogsDataset, self).__init__()

        self.datapath = datapath
        self.thres = thres
        self.device = device

        if split == 'random':
            pair_path = os.path.join(datapath, 'random_pairs.txt')
        elif split == 'class':
            pair_path = os.path.join(datapath, 'class_pairs.txt')
        else:
            raise Exception('Invalid pair type: %s' % split)
        
        with open(pair_path) as file:
            pairs = file.readlines()
            pairs = [pair.rstrip() for pair in pairs]
            if split == 'random':
                pairs = pairs[0:10000]
        
        self.src_imnames = [pair.split(' ')[0] for pair in pairs]
        self.trg_imnames = [pair.split(' ')[1] for pair in pairs]
        
        self.cls = os.listdir(os.path.join(datapath, 'Images'))
        self.cls.sort()

        anno_path = os.path.join(datapath, 'StanfordExtra_V12', 'StanfordExtra_v12.json')
        with open(anno_path) as f:
            json_data = json.load(f)
        self.annotations  = {i['img_path']: i for i in json_data}
        

    def __getitem__(self, idx):
        
        sample = dict() 
        
        sample['datalen'] = len(self.src_imnames)
       
        sample['src_imname'] = self.src_imnames[idx]
        sample['trg_imname'] = self.trg_imnames[idx]

        sample['src_img'] = self.get_image(self.src_imnames, idx)
        sample['trg_img'] = self.get_image(self.trg_imnames, idx)

        sample['src_bbox'] =  self.annotations[sample['src_imname']]['img_bbox']
        sample['trg_bbox'] =  self.annotations[sample['trg_imname']]['img_bbox']
        
        sample['pckthres'] = self.get_pckthres(sample).to(self.device)
        
        sample['src_kps'], sample['trg_kps'], sample['common_joints'] = self.get_kps(sample)
        sample['pair_class'] = sample['src_imname'].split('/')[0]

        return sample

    def __len__(self):
        return len(self.src_imnames)

    def get_image(self, img_names, idx):
        
        img_name = os.path.join(self.datapath, 'Images', img_names[idx])
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        return np.array(Image.open(path).convert('RGB'))

    def get_pckthres(self, sample):
        
        if self.thres == 'bbox':
            return torch.tensor(max(sample['trg_bbox'][2:]))
        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, sample):
        
        src_kps = np.array(self.annotations[sample['src_imname']]['joints'])
        trg_kps = np.array(self.annotations[sample['trg_imname']]['joints'])
        
        common_joints = np.where(src_kps[:,2] * trg_kps[:,2]==1)[0]
        return src_kps[common_joints,:2].T, trg_kps[common_joints,:2].T, common_joints





