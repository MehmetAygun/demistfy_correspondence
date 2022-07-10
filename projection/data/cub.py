import json
import glob
import os
import random
import pickle
import cv2
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import CorrespondenceDataset
from torch.utils.data import Dataset
from pycocotools.mask import decode as decode_RLE
from PIL import Image

import matplotlib.pyplot as plt
def get_seg_from_entry(entry):
    
    """Given a .json entry, returns the binary mask as a numpy array"""

    rle = {
            "size": [entry['img_height'], entry['img_width']],
            "counts": entry['seg']
    }

    decoded = decode_RLE(rle)
    return decoded

def get_bbox_from_seg(mask):

    idxs = np.where(mask==1)
    y0, x0 = np.min(idxs[0]), np.min(idxs[1])
    y1, x1 = np.max(idxs[0]), np.max(idxs[1])
    
    return torch.tensor([x0,y0,x1,y1])

class CUBDataset(Dataset):
    def __init__(self, datapath, transform, split='trn',augmentation=None, warper=None):
        
        super(CUBDataset, self).__init__()
        
        self.g_dim = 64
        self.datapath = datapath
        self.transform = transform
        self.split = split
        self.augmentation = augmentation
        self.warper = warper

        with open(os.path.join(datapath, 'train_test_split.txt')) as file:
            split = file.readlines()
            split = [s.rstrip().split(' ')[0] for s in split if s.strip().split(' ')[1] == '1']
       
        random.shuffle(split)
        n_train =int(len(split)*0.7)
        if split =='trn':
            split = split[0:n_train]
        else:
            split = split[n_train:]
        self.split = split
        with open(os.path.join(datapath, 'images.txt')) as file:
            images = file.readlines()
            images = [image.rstrip() for image in images]
            self.images = {image.split(' ')[0]:image.split(' ')[1] for image in images if image.split(' ')[0] in split}
        
        with open(os.path.join(datapath, 'nn_pairs.txt')) as file:
            pairs = file.readlines()
            pairs = [pair.rstrip() for pair in pairs]
            self.nn_pairs = {p.split(' ')[0]:p.split(' ')[1].split(',') for p in pairs if p.split(' ')[0] in split}
         
        self.keys = list(self.images.keys())
        with open(os.path.join(datapath, 'parts', 'part_locs.txt')) as file:
            parts = file.readlines()
            parts = [part.rstrip() for part in parts]
            self.kps = {part.split(' ')[0]: [] for part in parts}
            for part in parts:
                p = part.split(' ')
                kp = [float(c) for c in p[2:]]
                self.kps[p[0]].append(kp)

    def __getitem__(self, idx):
        
        sample = dict() 
        src_idx, trg_idx = random.sample(self.keys,2)
        
        #possible_trg_idxs = [t for t in self.nn_pairs[src_idx] if t in self.split]
        #if len(possible_trg_idxs)>0:
        #    trg_idx = random.sample(possible_trg_idxs,1)[0]
        
        sample['src_img'] = self.get_image(src_idx)
        sample['trg_img'] = self.get_image(trg_idx)
        
        sample['src_kps'], sample['trg_kps'] = self.get_kps(src_idx, trg_idx)
        
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
        

        #print (src_idx, trg_idx, sample['src_kps'], sample['trg_kps'])
        #print (s_H, s_W, src_xratio, src_yratio)
        #sys.exit()
        return sample

    def __len__(self):
        return 10000

    def get_image(self, idx):
        
        img_name = os.path.join(self.datapath, 'images', self.images[idx])
        image = self.get_imarr(img_name)
        #image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        img = cv2.imread(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if self.augmentation:
            img = self.augmentation(image=img)['image']
        img = img / 255.0

        return img


    def get_pckthres(self, sample):
        
        trg_bbox = sample['trg_bbox']
        if self.thres == 'bbox':
            return torch.tensor(max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1]))

        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, src_idx, trg_idx):
        
        src_kps = np.array(self.kps[src_idx])
        trg_kps = np.array(self.kps[trg_idx])
    
        src_kps[src_kps[:,2]==0] = -1
        trg_kps[trg_kps[:,2]==0] = -1

        return src_kps[:,:2], trg_kps[:,:2]



