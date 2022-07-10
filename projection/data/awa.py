import json
import glob
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from .dataset import CorrespondenceDataset
from torch.utils.data import Dataset
from pycocotools.mask import decode as decode_RLE
from PIL import Image

import cv2

class AWADataset(Dataset):
    def __init__(self, datapath, transform, split='trn',augmentation=None, warper=None):
        
        super(AWADataset, self).__init__()

        self.g_dim = 64
        self.datapath = datapath
        self.transform = transform
        self.split = split
        self.augmentation = augmentation
        self.warper = warper

        with open(os.path.join(self.datapath, 'train_images.txt')) as file:
            images = file.readlines()
            self.images = [im.rstrip() for im in images]
        
        self.data = {}
        for ann in self.images:
            with open(os.path.join(datapath, 'Annotations', ann.replace('.jpg','.pickle')), 'rb') as f:
                d = pickle.load(f)
            self.data[ann] = {} 
            parts = list(d['a1'].keys())
            parts = [p for p in parts if not p=='bbox']
            self.data[ann]['kps'] = {p:d['a1'][p] for p in parts}
                
    def __getitem__(self, idx):
        
        sample = dict() 
        
        src_idx, trg_idx = random.sample(self.images,2)
        
        sample['src_img'] = self.get_image(src_idx)
        sample['trg_img'] = self.get_image(trg_idx)
        
        sample['src_kps'], sample['trg_kps'] = self.get_kps(src_idx, trg_idx)
        
        sample['src_imname'] = src_idx
        sample['trg_imname'] = trg_idx

        sample['src_img'] = self.get_image(sample['src_imname'])
        sample['trg_img'] = self.get_image(sample['trg_imname'])
       
        src_kps, trg_kps = self.get_kps(src_idx, trg_idx)
        
        sample['src_kps'] = np.ones((40,2)) * -1
        sample['trg_kps'] = np.ones((40,2)) * -1
        
        n_k = src_kps.shape[0]
        sample['src_kps'][:n_k,:] = src_kps
        sample['trg_kps'][:n_k,:] = trg_kps

        s_H, s_W = sample['src_img'].shape[:2]
        t_H, t_W = sample['trg_img'].shape[:2]
   
        H, W = self.g_dim, self.g_dim
        src_xratio = W*1./ s_W
        src_yratio = H*1./ s_H

        trg_xratio = W*1./ t_W
        trg_yratio = H*1./ t_H
 
        sample['src_kps'][:,0] = sample['src_kps'][:,0] * src_xratio
        sample['src_kps'][:,1] = sample['src_kps'][:,1] * src_yratio
        
        sample['trg_kps'][:,0] = sample['trg_kps'][:,0] * trg_xratio
        sample['trg_kps'][:,1] = sample['trg_kps'][:,1] * trg_yratio

        sample['src_kps'] = torch.tensor(sample['src_kps'])
        sample['trg_kps'] = torch.tensor(sample['trg_kps'])
        
        sample['src_kps'] = torch.min(sample['src_kps'], torch.ones_like(sample['src_kps'])*(self.g_dim-1))
        sample['trg_kps'] = torch.min(sample['trg_kps'], torch.ones_like(sample['trg_kps'])*(self.g_dim-1))
       
        if self.warper:
            im1 = torch.from_numpy(sample['src_img']) * 255
            im1, im2, flow, grid, kp2, kp1 = self.warper(im1.permute(2,0,1).float())
            sample['src_img1'] = self.transform({'image':im1.permute(1,2,0).numpy()/255.})['image']
            sample['src_img2'] = self.transform({'image':im2.permute(1,2,0).numpy()/255.})['image']
            sample['src_grid'] = grid.squeeze()
            sample['src_flow'] = flow
            
            im1 = torch.from_numpy(sample['trg_img']) * 255
            im1, im2, flow, grid, kp2, kp1 = self.warper(im1.permute(2,0,1).float())
            sample['trg_img1'] = self.transform({'image':im1.permute(1,2,0).numpy()/255.})['image']
            sample['trg_img2'] = self.transform({'image':im2.permute(1,2,0).numpy()/255.})['image']
            sample['trg_grid'] = grid.squeeze()
            sample['trg_flow'] = flow



        sample['src_img'] = self.transform({'image':sample['src_img']})['image']
        sample['trg_img'] = self.transform({'image':sample['trg_img']})['image']
        
        return sample

    def __len__(self):
        return 5000

    def get_image(self, src_idx):
        
        img_name = os.path.join(self.datapath, 'JPEGImages', src_idx)
        image = self.get_imarr(img_name)
        #image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        img = cv2.imread(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        return img
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

    def get_kps(self, src_idx, trg_idx):
        
        
        src_kps = []
        trg_kps = []
        for kp, v in self.data[src_idx]['kps'].items():
            if kp in self.data[trg_idx]['kps']:
                src_kps.append(v) 
                trg_kps.append(self.data[trg_idx]['kps'][kp])

        src_kps = np.array(src_kps)
        trg_kps = np.array(trg_kps)
        
        common_joints = np.where(np.logical_and(src_kps[:,0]!=-1, trg_kps[:,1]!=-1))[0]
        return src_kps[common_joints,:2], trg_kps[common_joints,:2]





