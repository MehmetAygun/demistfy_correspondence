import argparse
import datetime
import os
import pickle
import timeit
import numpy as np 
import math

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import cv2
import sys
sys.path.append('../')
sys.path.append('../../')
from model import  geometry, evaluation, util
from data import stanforddogs


from torchvision.transforms import Compose
from backbone.models import FeatureNet
from backbone.transforms import Resize, NormalizeImage, PrepareForNet
from backbone.transformer import ViTExtractor

from util.helper import vit_feat_extract, read_image, find_dim_weights, pred_kps


def run(datapath, split, benchmark, featurepath, thres, alpha, logpath, layers, w, model_type, model, transform, visualize=False):

    # 1. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    
    cur_datetime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    logfile = os.path.join('logs', logpath + cur_datetime + '.log')
    util.init_logger(logfile)
    util.log_args(args)
    if visualize: os.mkdir(logfile + 'vis')

    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if w:
        w= w.to(device)
    if model_type != 'vit' and model_type !='dino': 
        model = model.to(device)

    dset = stanforddogs.StanfordDogsDataset(datapath, thres, device, split)
    dataloader = DataLoader(dset, batch_size=1, num_workers=0)
    
    #3. Evaluator initialization
    evaluator = evaluation.Evaluator(benchmark, device)

    for idx, data in enumerate(dataloader):
        t1 = timeit.default_timer()
 
        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
        data['alpha'] = alpha
        t2 = timeit.default_timer()
        src_img_path = os.path.join(datapath, 'Images', data['src_imname'][0])
        trg_img_path = os.path.join(datapath, 'Images', data['trg_imname'][0])
        
        if model_type == 'vit' or model_type == 'dino':
            src_feature = vit_feat_extract(model, device, src_img_path)
            trg_feature = vit_feat_extract(model, device, trg_img_path)
     
        else:
            src_im = torch.from_numpy(transform({"image": read_image(src_img_path)})["image"]).unsqueeze(0)
            trg_im = torch.from_numpy(transform({"image": read_image(trg_img_path)})["image"]).unsqueeze(0)
             
            inp = torch.cat([src_im, trg_im], dim=0).to(device)
            features = model(inp)
            src_feature, trg_feature = [f[0].unsqueeze(0) for f in features], [f[1].unsqueeze(0) for f in features],        
        
        t3 = timeit.default_timer()
 
        
        # c) Predict key-points & evaluate performance
        prd_kps = pred_kps(data['src_img'], data['trg_img'], src_feature, trg_feature,  data['src_kps'], w=w, layers=layers)
        
        t4 = timeit.default_timer()
        evaluator.evaluate(prd_kps.to(device), data)
        t5 = timeit.default_timer()
 
        # d) Log results
        
        evaluator.log_result(idx, data=data)
        if visualize:
            vispath = os.path.join(logfile + 'vis', '%03d_%s_%s' % (idx, data['src_imname'][0], data['trg_imname'][0]))
            util.visualize_prediction(data['src_kps'].t().cpu(), prd_kps.t().cpu(),
                                      data['src_img'], data['trg_img'], vispath)
        t6 = timeit.default_timer()
        """
        print ('data load {}'.format(t2-t1))
        print ('fet load {}'.format(t3-t2))
        print ('kp pred {}'.format(t4-t3))
        print ('kp eval {}'.format(t5-t4))
        print ('logging  {}'.format(t6-t5))
        """

    evaluator.log_result(len(dset), data=None, average=True)


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Hyperpixel Flow in pytorch')
    parser.add_argument('--datapath', type=str, default='/home/s2254720/data/StanfordDogs')
    parser.add_argument('--featurepath', type=str, default='/data1/StanfordDogs/features_resnet50/')
    parser.add_argument('--benchmark', type=str, default='stanforddog')
    parser.add_argument('--thres', type=str, default='bbox', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--layers', type=str, default='2,3,4')
    parser.add_argument('--split', type=str, default='random')
    
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--model_type', type=str,default='resnet50')
    
    parser.add_argument('--projpath', type=str, default='') 
    parser.add_argument('--dimredpath', type=str, default='') 
    
    args = parser.parse_args()
    args.layers = args.layers.split(',')
    
    if args.projpath != '':
        if args.projpath == 'random':
            if args.model_type == 'vit':
                pdim = 768
            elif args.model_type == 'dino':
                pdim = 384
            else:
                pdim = 1024
            w = torch.nn.Sequential(
                torch.nn.Conv2d(pdim, 256, (1,1)),
                )
        else:
            w = torch.load(args.projpath)
    else:
        w = None
    
    if args.dimredpath:
        if args.model_type == 'vit':
            pdim = 768
        elif args.model_type == 'dino':
            pdim = 384
        else:
            pdim = 1024

        w = torch.nn.Sequential(
            torch.nn.Conv2d(pdim, 256, (1,1)),
            )

        import pickle
        with open (args.dimredpath,'rb') as f:
            dr = pickle.load(f)
        w[0].weight = torch.nn.Parameter(torch.from_numpy(dr.components_).view(256,pdim,1,1))



    model_type = args.model_type
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
        normalization = None
    elif model_type == 'dino':
        m = 'dino_vits8' 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stride = 8
        model  = ViTExtractor(m, stride, device=device)
        net_w, net_h = 224, 224
        normalization = None
 
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

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
    
    if model_type != 'vit' and model_type != 'dino':
        model.eval()
    
    run(datapath=args.datapath, split=args.split, benchmark = args.benchmark, 
        featurepath=args.featurepath, thres=args.thres, alpha=args.alpha, 
        logpath=args.logpath, visualize=args.visualize, layers=args.layers,
        w=w, model_type=model_type, model=model, transform=transform)
