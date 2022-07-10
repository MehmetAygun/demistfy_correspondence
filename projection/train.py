import argparse
import datetime
import os
import pickle
import sys
import math 
import time

import numpy as np 
sys.path.append('../')
import cv2
import albumentations as A

from torchvision.transforms import Compose
from backbone.models import FeatureNet
from backbone.transforms import Resize, NormalizeImage, PrepareForNet, Crop
from backbone.transformer import ViTExtractor
from loss import *

import torchvision.models as models
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from data import spair, cub, stanforddogs, awa, aflw
from model import util 

import logging
import matplotlib.pyplot as plt
from matplotlib import cm

import tps


def vit_feat_extract(extractor, img_batch, facet='key', binning=False):
    
    layers = [2,5,9,11]
    descs = extractor.extract_descriptors(img_batch, layers, facet, binning)
    feat = []
    for l_idx, feature in enumerate(layers):
        desc = descs[l_idx]
        desc = desc.squeeze().permute(0,2,1)
        b, d, s = desc.shape
        h = int(math.sqrt(s))
        desc = desc.reshape(b, d, h ,h)
        feat.append(desc)
    return feat


def run(datapath, benchmark, logpath, layers, batchsize, w, optim, model_type, model, transform, augmentation, temps, dve, cl, lead, lead_mse, asym, asym_ce, eq, in_ratio, epoch=51, gdim=64):
    
    # 1. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    
    cur_datetime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    logfile = os.path.join('logs', logpath + cur_datetime + '.log')

    util.init_logger(logfile)
    util.log_args(args)

    # 2. Data initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dve or eq:
        trn_warper = tps.Warper(384, 384)
        val_warper = tps.Warper(384, 384)
    else:
        trn_warper = None
        val_warper = None

    w = w.to(device)
    if model_type != 'vit' and model_type!='dino':
        model = model.to(device)
    t1, t2 = temps
    if benchmark == 'spair':
        train_dset = spair.SPairDataset(benchmark, datapath, transform, 'trn', augmentation, trn_warper)
        val_dset = spair.SPairDataset(benchmark, datapath, transform, 'val', augmentation, val_warper)
    
    elif benchmark == 'cub':
        train_dset = cub.CUBDataset(datapath, transform, 'trn', augmentation, trn_warper)
        val_dset = cub.CUBDataset(datapath, transform, 'val', augmentation, val_warper)

    elif benchmark == 'stanforddogs':
        train_dset = stanforddogs.StanfordDogsDataset(datapath, transform, 'trn', augmentation, trn_warper)
        val_dset = stanforddogs.StanfordDogsDataset(datapath, transform, 'val', augmentation, val_warper)
    
    elif benchmark == 'awa':
        train_dset = awa.AWADataset(datapath, transform, 'trn', augmentation, trn_warper)
        val_dset = awa.AWADataset(datapath, transform, 'val', augmentation, val_warper)
    
    elif benchmark == 'aflw':
        train_dset = aflw.AFLWDataset(datapath, transform, 'trn', augmentation, trn_warper)
        val_dset = aflw.AFLWDataset(datapath, transform, 'val', augmentation, val_warper)
    
    elif benchmark == 'celeba':
        train_dset = aflw.CelebaDataset(datapath, transform, 'trn', augmentation, trn_warper)
        val_dset = aflw.CelebaDataset(datapath, transform, 'val', augmentation, val_warper)
 
    else:
        assert 'wrong dataset'
    if benchmark=='celeba' or benchmark == 'aflw':
        dataloader_train = DataLoader(train_dset, batch_size=batchsize, num_workers=0, shuffle=True)
        dataloader_test = DataLoader(val_dset, batch_size=batchsize, num_workers=0)
    else:
        dataloader_train = DataLoader(train_dset, batch_size=batchsize, num_workers=4, shuffle=True)
        dataloader_test = DataLoader(val_dset, batch_size=batchsize, num_workers=2)
 
    if dve or eq:
        gdim = 32
    if benchmark == 'celeba':
        gdim = 48
    
    best_val_loss = 666
    for e in range(epoch):
        for idx, data in enumerate(dataloader_train):
            if idx == 1000:
                break

            if dve or eq:
                with torch.no_grad():
                    imgs = torch.cat([data['src_img1'], data['src_img2'], data['trg_img1'], data['trg_img2']],dim=0)
                    if model_type =='dino' or model_type =='vit':
                        feats = vit_feat_extract(model, imgs.to(device))
                    else:
                        feats = model(imgs.to(device))
     
                grid = torch.cat([data['src_grid'], data['trg_grid']],dim=0)

                feats = [F.interpolate(feats[int(i)-1], size=(gdim, gdim), mode='bilinear', align_corners=False) for i in layers]
                feats = torch.cat(feats, dim=1) 
                proj_feats = w(feats)
                if eq:
                    loss = dense_correlation_loss(proj_feats,{'grid':grid})  
                else:
                    loss = dense_correlation_loss_dve(proj_feats,{'grid':grid, 't':t1}) 
                optim.zero_grad() 
                loss.backward()
                optim.step()
                if idx % 25 == 0:
                    logging.info ('Epoch-{} Iter-{} Loss:{}'.format(e, idx, loss.item()))
                continue

            else:
                with torch.no_grad():
                    if model_type =='dino' or model_type =='vit':
                        #tick1 = time.time()
                        inp_d = torch.cat([data['src_img'].to(device), data['trg_img'].to(device)],dim=0)
                        feats = vit_feat_extract(model, inp_d)
                        b,_,_,_ = inp_d.shape
                        src_f, trg_f = [f[0:b//2] for f in feats], [f[b//2:] for f in feats]
                    else:
                        inp_d = torch.cat([data['src_img'].to(device), data['trg_img'].to(device)],dim=0)
                        feats = model(inp_d)
                        b,_,_,_ = inp_d.shape
                        src_f, trg_f = [f[0:b//2] for f in feats], [f[b//2:] for f in feats]

    
                    src_feat = [F.interpolate(src_f[int(i)-1], size=(gdim, gdim), mode='bilinear', align_corners=False) for i in layers]
                    trg_feat = [F.interpolate(trg_f[int(i)-1], size=(gdim, gdim), mode='bilinear', align_corners=False) for i in layers]
                    src_feat = torch.cat(src_feat, dim=1) 
                    trg_feat = torch.cat(trg_feat, dim=1) 
                    
                proj = w(torch.cat([src_feat, trg_feat],dim=0))
                b = proj.shape[0]
                src_proj, trg_proj = proj[0:b//2], proj[b//2:]
            
            optim.zero_grad() 
            if cl:
                loss = cl_loss(torch.cat([src_proj,trg_proj],dim=0), torch.cat([data['src_img'], data['trg_img']], dim=0).size(), t1, [256])
            elif lead:
                loss = lead_loss(src_feat, trg_feat, src_proj, trg_proj, t1)
            elif lead_mse:
                loss = lead_mse_loss(src_feat, trg_feat, src_proj, trg_proj, t1)
            elif asym_ce:
                loss = asym_ce_loss(src_feat, trg_feat, src_proj, trg_proj, t1, t2)
            elif asym:
                loss = asym_loss(src_feat, trg_feat, src_proj, trg_proj, t1, t2)
            else:
                loss = sup_loss(src_proj, trg_proj, data['src_kps'].to(device), data['trg_kps'].to(device))
            
            loss.backward()
            optim.step()
            if idx % 25 == 0:
                logging.info ('Epoch-{} Iter-{} Loss:{}'.format(e, idx, loss.item()))
        
        if e % 10 == 0:
            savefile = os.path.join('ckpts', logpath + cur_datetime + '_ep{}.log'.format(e))
            torch.save(w, savefile)
   
        if e % 10 == 0:
            val_loss = 0.
            for idx, data in enumerate(dataloader_test):
                if idx == 250:
                    break
                if dve or eq:
                    with torch.no_grad():
                        imgs = torch.cat([data['src_img1'], data['src_img2'], data['trg_img1'], data['trg_img2']],dim=0)
                        if model_type =='dino' or model_type =='vit':
                            feats = vit_feat_extract(model, imgs.to(device))
                        else:
                            feats = model(imgs.to(device))
     
                        grid = torch.cat([data['src_grid'], data['trg_grid']],dim=0)

                        feats = [F.interpolate(feats[int(i)-1], size=(gdim, gdim), mode='bilinear', align_corners=False) for i in layers]
                        feats = torch.cat(feats, dim=1) 
                        proj_feats = w(feats)
                        if eq:
                            loss = dense_correlation_loss(proj_feats,{'grid':grid})  
                        else:
                            loss = dense_correlation_loss_dve(proj_feats,{'grid':grid, 't':t1}) 
                        val_loss += loss.item()

                else:
                    with torch.no_grad():
                        if model_type =='dino' or model_type =='vit':
                            inp_d = torch.cat([data['src_img'].to(device), data['trg_img'].to(device)],dim=0)
                            feats =  vit_feat_extract(model, inp_d)
                            b,_,_,_ = inp_d.shape
                            src_f, trg_f = [f[0:b//2] for f in feats], [f[b//2:] for f in feats]

                        else:
                            inp_d = torch.cat([data['src_img'].to(device), data['trg_img'].to(device)],dim=0)
                            feats = model(inp_d)
                            b,_,_,_ = inp_d.shape
                            src_f, trg_f = [f[0:b//2] for f in feats], [f[b//2:] for f in feats]

                        src_feat = [F.interpolate(src_f[int(i)-1], size=(gdim, gdim), mode='bilinear', align_corners=False) for i in layers]
                        trg_feat = [F.interpolate(trg_f[int(i)-1], size=(gdim, gdim), mode='bilinear', align_corners=False) for i in layers]
                        src_feat = torch.cat(src_feat, dim=1) 
                        trg_feat = torch.cat(trg_feat, dim=1) 

                        proj = w(torch.cat([src_feat, trg_feat],dim=0))
                        b = proj.shape[0]
                        src_proj, trg_proj = proj[0:b//2], proj[b//2:]

                        if cl:
                            loss = cl_loss(torch.cat([src_proj,trg_proj],dim=0), torch.cat([data['src_img'], data['trg_img']], dim=0).size(), t1, [256])
                        elif lead:
                            loss = lead_loss(src_feat, trg_feat, src_proj, trg_proj, t1)
                        elif lead_mse:
                            loss = lead_mse_loss(src_feat, trg_feat, src_proj, trg_proj, t1)
                        elif asym_ce:
                            loss = asym_ce_loss(src_feat, trg_feat, src_proj, trg_proj, t1, t2)
                        elif asym:
                            loss = asym_loss(src_feat, trg_feat, src_proj, trg_proj, t1, t2)
                        else:
                            loss = sup_loss(src_proj, trg_proj, data['src_kps'].to(device), data['trg_kps'].to(device))
                        
                        val_loss += loss.item()
            logging.info ('Epoch-{} Val loss:{}'.format(e, val_loss/len(dataloader_test)))
            if val_loss/len(dataloader_test) < best_val_loss:
                best_val_loss = val_loss/len(dataloader_test)
                savefile = os.path.join('ckpts', logpath + '_best.log')
                torch.save(w, savefile)
 
if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Unsup Semantic Correspondence Trainings')
    parser.add_argument('--datapath', type=str, default='/home/s2254720/data/')
    parser.add_argument('--dataset', type=str, default='spair')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--layers', type=str, default='3')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--weightdecay', type=float, default=1e-3)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--model_type', type=str,default='resnet50')
    parser.add_argument('--useaugmentation', action='store_true')
    
    parser.add_argument('--dve', action='store_true')
    parser.add_argument('--eq', action='store_true')
 
    parser.add_argument('--t1', type=float, default=10)
    parser.add_argument('--t2', type=float, default=5)
    parser.add_argument('--cl', action = 'store_true')
    parser.add_argument('--lead', action = 'store_true')
    parser.add_argument('--lead_mse', action = 'store_true')
    parser.add_argument('--asym_ce', action = 'store_true')
    parser.add_argument('--asym', action = 'store_true')


    parser.add_argument('--in_ratio', type=float, default=1.0)
    
   
    args = parser.parse_args()
    args.layers = args.layers.split(',')
    
    temps = [args.t1, args.t2]

    if args.useaugmentation:
        augmentation = A.Compose([
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2)
            ])
    else:
        augmentation = None

    model_type = args.model_type
    batch_size = args.batchsize
    model_path = args.model_weights
   
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if model_type == "resnet50": #resnet50
        model = FeatureNet(
            backbone="resnet50", path=args.model_weights
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    elif model_type == "inat_resnet50": #resnet50
        model = FeatureNet(
            backbone="inat_resnet50", path=args.model_weights
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif model_type == "coco_resnet50": #resnet50
        model = FeatureNet(
            backbone="coco_resnet50", path=args.model_weights
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif model_type == "convnext": #resnet50
        model = FeatureNet(
            backbone="convnext"
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif model_type == 'vit':
        m = 'vit_base_patch8_224' 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stride = 8
        model  = ViTExtractor(m, stride, device=device)
        net_w, net_h = 224, 224
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == 'dino':
        m = 'dino_vits8' 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stride = 8
        model  = ViTExtractor(m, stride, device=device)
        net_w, net_h = 224, 224
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    if model_type != 'vit' and model_type != 'dino':
        model.eval()
        d = [256,512,1024,2048]
    else:
        if model_type == 'vit':
            d =[768, 768, 768, 768]
        else:
            d =[384, 384, 384, 384]

    
    warper = None
    if args.eq or args.dve:
        warper = tps.Warper(net_h, net_w)
    if args.dataset == 'celeba':
        net_w,net_h = 136,136

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    if args.dataset == 'celeba':
        transform = Compose(
            
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    image_interpolation_method=cv2.INTER_CUBIC
                ),
                Crop(20),
                normalization,
                PrepareForNet(),
            ]
        )


    in_dim = sum([d[int(l)-1] for l in args.layers]) 
    
    w = torch.nn.Sequential(
        torch.nn.Conv2d(in_dim, args.dim, (1,1)),
    )
    
    optim = torch.optim.Adam(w.parameters(), lr=0.001, weight_decay=args.weightdecay)
 
    run(datapath=args.datapath, benchmark=args.dataset,
        logpath=args.logpath, layers=args.layers, batchsize=args.batchsize, w=w, optim=optim, model_type= model_type, model=model, transform=transform, 
        augmentation=augmentation, dve=args.dve, temps=temps, eq= args.eq, cl=args.cl, lead= args.lead, lead_mse=args.lead_mse, asym = args.asym, asym_ce=args.asym_ce, in_ratio=args.in_ratio)
