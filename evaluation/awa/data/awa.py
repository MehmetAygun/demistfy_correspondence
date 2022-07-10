import json
import glob
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class AWADataset(Dataset):
    def __init__(self, datapath, thres, device, split='random'):
        
        super(AWADataset, self).__init__()

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
        
        self.cls = os.listdir(os.path.join(datapath,'Annotations'))
        
        self.data = {}
        for cls in self.cls:
            cls_annotations = os.listdir(os.path.join(datapath, 'Annotations', cls))
            for ann in cls_annotations:
                with open(os.path.join(datapath, 'Annotations', cls, ann), 'rb') as f:
                    d = pickle.load(f)
                key = '{}/{}'.format(cls, ann.replace('pickle','jpg'))
                self.data[key] = {}
                
                parts = list(d['a1'].keys())
                parts = [p for p in parts if not p=='bbox']
                self.data[key]['kps'] = {p:d['a1'][p] for p in parts}
                
                if not 'bbox' in d['a1']:
                    self.data[key]['bbox'] = [0,0,0,0]
                else:
                    self.data[key]['bbox'] = d['a1']['bbox']

    def __getitem__(self, idx):
        
        sample = dict() 
        
        sample['datalen'] = len(self.pairs)
       
        sample['src_imname'] = self.pairs[idx][0]
        sample['trg_imname'] = self.pairs[idx][1]

        sample['src_img'] = self.get_image(sample['src_imname'])
        sample['trg_img'] = self.get_image(sample['trg_imname'])

        sample['src_bbox'] = self.data[sample['src_imname']]['bbox']
        sample['trg_bbox'] = self.data[sample['trg_imname']]['bbox']

        sample['pckthres'] = self.get_pckthres(sample).to(self.device)
        
        sample['src_kps'], sample['trg_kps'], sample['common_joints'] = self.get_kps(sample)
        sample['pair_class'] = sample['src_imname'].split('/')[0]

        return sample

    def __len__(self):
        return len(self.pairs)

    def get_image(self, img_name):
        
        img_name = os.path.join(self.datapath, 'JPEGImages', img_name)
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        return np.array(Image.open(path).convert('RGB'))

    def get_pckthres(self, sample):
        
        if self.thres == 'bbox':
            if (sample['trg_bbox'])!=0:
                trg_bbox = sample['trg_bbox']
                return torch.tensor(max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1]))
            else:
                return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))

        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, sample):
        
        src_k = list(self.data[sample['src_imname']]['kps'].keys())
        #trg_kps = list(self.data[sample['trg_imname']]['kps'].keys())
        
        #common_kps = np.intersect1d(src_kps, trg_kps)
        src_kps = []
        trg_kps = []
        joints = []
        for kp in src_k:
            if kp in self.data[sample['trg_imname']]['kps']:
                src_kps.append(self.data[sample['src_imname']]['kps'][kp]) 
                trg_kps.append(self.data[sample['trg_imname']]['kps'][kp])
                joints.append(kp)
        src_kps = np.array(src_kps)
        trg_kps = np.array(trg_kps)
        
        common_joints = np.where(np.logical_and(src_kps[:,0]!=-1, trg_kps[:,1]!=-1))[0]
        return src_kps[common_joints,:2].T, trg_kps[common_joints,:2].T, joints





