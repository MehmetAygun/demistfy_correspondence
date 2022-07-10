import json
import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torchvision
from .dataset import CorrespondenceDataset
from torch.utils.data import Dataset
from pycocotools.mask import decode as decode_RLE
from PIL import Image

import matplotlib.pyplot as plt

class CelebaDataset(Dataset):

    def __init__(self, datapath, transform, split='trn',augmentation=None, warper=None):
        
        super(CelebaDataset, self).__init__()

        self.datapath = datapath
        self.transform = transform
        self.g_dim = 64
        self.augmentation = augmentation
        self.warper = warper
        val_size = 2000
        self.data_split = split

        anno = pd.read_csv(os.path.join(datapath, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
            delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(datapath, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(datapath, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(datapath, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        # Ensure that we are not using mafl images
        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 4).sum() == 1000
        
        if self.data_split == 'trn':
            self.data = anno.loc[split[split[1] == 0].index]
        else:
            # subsample images from CelebA val, otherwise training gets slow
            self.data = anno.loc[split[split[1] == 2].index][:val_size]
        
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)
    
        # Move head up a bit
        initial_crop = lambda im: torchvision.transforms.functional.crop(im, 30, 0, 178, 178)
        self.keypoints[:, :, 1] -= 30
        
        self.initial_transforms = torchvision.transforms.Compose([initial_crop])


    def __getitem__(self, idx):
        
        sample = dict() 
        
        src_idx, trg_idx = np.random.randint(0,len(self.filenames),2)
        
        sample['src_img'] = self.get_image(src_idx)[30:,:]
        sample['trg_img'] = self.get_image(trg_idx)[30:,:]
        
        sample['src_imname'] = self.filenames[src_idx]
        sample['trg_imname'] = self.filenames[trg_idx]
       
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
 
        return sample

    def __len__(self):
        return 5000

    def get_image(self, src_idx):
        
        img_name = os.path.join(self.datapath, 'Img/img_align_celeba', self.filenames[src_idx])
        image = self.get_imarr(img_name)

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

    def get_kps(self, src_idx, trg_idx):
        
        src_kps = self.keypoints[src_idx]
        trg_kps = self.keypoints[trg_idx]
 
        src_kps = np.array(src_kps)
        trg_kps = np.array(trg_kps)

        return src_kps, trg_kps


class AFLWDataset(Dataset):
    def __init__(self, datapath, transform, split='trn',augmentation=None, warper=None):
        
        super(AFLWDataset, self).__init__()

        self.datapath = datapath
        self.transform = transform
        self.g_dim = 64
        self.augmentation = augmentation
        self.warper = warper

        with open(os.path.join(datapath, 'training.txt')) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        
        random.shuffle(lines)
        n = int(len(lines)*0.7)
        if split == 'trn':
            lines = lines[0:n]
        else:
            lines = lines[n:]

        self.data = {}
        self.images = []
        for idx, line in enumerate(lines):
            l = line.split(' ')
            self.data[l[1]] = {}
            self.images.append(l[1])
            self.data[l[1]]['kps'] = []
            for x, y in zip(l[2:7],l[7:12]):
                self.data[l[1]]['kps'].append([float(x), float(y)])
        

    def __getitem__(self, idx):
        
        sample = dict() 
        
        src_idx, trg_idx = random.sample(self.images,2)
        
        sample['src_img'] = self.get_image(src_idx)
        sample['trg_img'] = self.get_image(trg_idx)
        
        sample['src_imname'] = src_idx
        sample['trg_imname'] = trg_idx
       
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
 
        return sample

    def __len__(self):
        return 5000

    def get_image(self, src_idx):
        
        img_name = os.path.join(self.datapath, src_idx)
        image = self.get_imarr(img_name)
        #image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        img = cv2.imread(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        H,W,C = img.shape
        c_h, c_w = int(H*0.25), int(W*0.25)
        if c_h < 1:
            c_h, c_w = H, W 
        img = img[c_h:-c_h,c_w:-c_w,:]

        return img

    def get_pckthres(self, sample):
        
        if self.thres == 'bbox':
            trg_bbox = sample['trg_bbox'] 
            return torch.tensor(max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1]))

        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_kps(self, src_idx, trg_idx):
        
        src_kps = self.data[src_idx]['kps']
        trg_kps = self.data[trg_idx]['kps']
 
        src_kps = np.array(src_kps)
        trg_kps = np.array(trg_kps)

        return src_kps, trg_kps



