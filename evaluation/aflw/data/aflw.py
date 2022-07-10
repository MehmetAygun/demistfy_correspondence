import json
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class AFLWDataset(Dataset):
    def __init__(self, datapath, thres, device, split='random'):
        
        super(AFLWDataset, self).__init__()

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
            pairs = [pair.rstrip().split(' ') for pair in pairs]
            self.pairs = pairs[0:10000]
        
        with open(os.path.join(datapath, 'test_classes.txt')) as file:
            lines = file.readlines()
            self.classes = [line.rstrip() for line in lines]
        
        with open(os.path.join(datapath, 'testing.txt')) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        
        self.data = {}
        for idx, line in enumerate(lines):
            l = line.split(' ')
            self.data[l[1]] = {}
            self.data[l[1]]['kps'] = []
            self.data[l[1]]['cls'] = self.classes[idx]
            for x, y in zip(l[2:7],l[7:12]):
                self.data[l[1]]['kps'].append([float(x), float(y)])
        
        self.cls = [str(t) for t in range(40)]

    def __getitem__(self, idx):
        
        sample = dict() 
        
        sample['datalen'] = len(self.pairs)
       
        sample['src_imname'] = self.pairs[idx][0]
        sample['trg_imname'] = self.pairs[idx][1]

        sample['src_img'] = self.get_image(sample['src_imname'])
        sample['trg_img'] = self.get_image(sample['trg_imname'])

        #sample['src_bbox'] =  None
        #sample['trg_bbox'] =  None
        
        sample['pckthres'] = self.get_pckthres(sample).to(self.device)
        
        sample['src_kps'], sample['trg_kps'] = self.get_kps(sample)
        sample['common_joints'] = np.arange(sample['src_kps'].shape[1])
        sample['pair_class'] = self.data[sample['src_imname']]['cls']

        return sample

    def __len__(self):
        return len(self.pairs)

    def get_image(self, img_name):
        
        img_name = os.path.join(self.datapath, img_name)
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        return np.array(Image.open(path).convert('RGB'))

    def get_pckthres(self, sample):
        
        if self.thres == 'bbox':
            trg_bbox = sample['trg_bbox'] 
            return torch.tensor(max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1]))

        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, sample):
        
        src_kps = self.data[sample['src_imname']]['kps']
        trg_kps = self.data[sample['trg_imname']]['kps']
 
        src_kps = np.array(src_kps)
        trg_kps = np.array(trg_kps)

        return src_kps.T, trg_kps.T



