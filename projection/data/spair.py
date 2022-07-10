r"""SPair-71k dataset"""
import json
import glob
import os
import pickle

import numpy as np
import torch

from .dataset import CorrespondenceDataset
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt

class SPairDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, transform, split, augmentation=None, warper=None, thres=0):
        r"""SPair-71k dataset constructor"""
        super(SPairDataset, self).__init__(benchmark, datapath, thres, split)
        
        self.g_dim = 64
        self.augmentation = augmentation
        self.transform = transform
        self.warper = warper
        
        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()
        
        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        anntn_files = list(map(lambda x: json.load(open(x)), anntn_files))
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']), anntn_files))
        self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']), anntn_files))
        self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))
        
        #self.features = {}
        """
        for cls in self.cls:
            self.features[cls] = {}
        for data_name in self.train_data:
            src = data_name.split('-')[1]
            trg, cls_name = data_name.split('-')[2].split(':')
            for s in [src,trg]:
                if not s in self.features[cls_name]:
                    ft_path = os.path.join(featurepath, cls_name, s) 
                    with open(ft_path, 'rb') as f:
                        src_feat = pickle.load(f)
                     self.features[cls_name][s] = src_feat
        """

    def __getitem__(self, idx):
        r"""Construct and return a batch for SPair-71k dataset"""
        sample = super(SPairDataset, self).__getitem__(idx)
        
        sample['src_img'] = self.get_image(self.src_imnames, idx)
        sample['trg_img'] = self.get_image(self.trg_imnames, idx)
 
        s_H, s_W = sample['src_img'].shape[:2]
        t_H, t_W = sample['trg_img'].shape[:2]
         
        H, W = self.g_dim, self.g_dim
        src_xratio = W*1./ s_W
        src_yratio = H*1./ s_H

        trg_xratio = W*1./ t_W
        trg_yratio = H*1./ t_H
        
        src_kps = torch.full((30,2),-1)
        trg_kps = torch.full((30,2),-1)
      
        n = sample['src_kps'].shape[0]#batch expect same size of kps
        trg_kps[:n,:] = sample['trg_kps']
        
        src_kps[:n,0] = sample['src_kps'][:,0] * src_xratio
        src_kps[:n,1] = sample['src_kps'][:,1] * src_yratio
        
        trg_kps[:n,0] = sample['trg_kps'][:,0] * trg_xratio
        trg_kps[:n,1] = sample['trg_kps'][:,1] * trg_yratio
        
        sample['src_kps'] = src_kps
        sample['trg_kps'] = trg_kps
 
        sample['src_bbox'] = self.src_bbox[idx]
        sample['trg_bbox'] = self.trg_bbox[idx]

        sample['vpvar'] = self.vpvar[idx]
        sample['scvar'] = self.scvar[idx]
        sample['trncn'] = self.trncn[idx]
        sample['occln'] = self.occln[idx]
 
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

    def get_image(self, img_names, idx):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
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



    def get_points(self, pts, idx):
        r"""Return key-points of an image"""
        return super(SPairDataset, self).get_points(pts, idx)
