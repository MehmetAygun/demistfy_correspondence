from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import cv2
import numpy as np


def vit_feat_extract(extractor, device, img_name, load_size=224, facet='key', binning=False):
    
    layers = [2,5,9,11]
    t1 = timeit.default_timer() 
 
    image_batch, image_pil = extractor.preprocess(img_name, load_size)
    
    descs = extractor.extract_descriptors(image_batch.to(device), layers, facet, binning)
    feat = []
    for l_idx, feature in enumerate(layers):
        desc = descs[l_idx]
        desc = desc.squeeze().T
        d, s = desc.shape
        h = int(math.sqrt(s))
        desc = desc.reshape(1, d, h ,h)
        feat.append(desc)
    t2 = timeit.default_timer() 
 
    return feat

def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def find_dim_weights(q, features):
    D, H, W = features.size()
    m_q = torch.mean(features.view(D, H*W), 1)
    diff = torch.pow(torch.pow((q - m_q),2),0.5)
    diff = F.normalize(diff,p=1,dim=0)
    return diff


def pred_kps(src_im, trg_im, src_feat, trg_feat, src_kps, layers=[1,4], w=None, bbox=None, grid_dim=128):
    
    src_feats = [F.interpolate(src_feat[int(i)-1], size=(grid_dim, grid_dim), mode='bilinear', align_corners=False) for i in layers]
    trg_feats = [F.interpolate(trg_feat[int(i)-1], size=(grid_dim, grid_dim), mode='bilinear', align_corners=False) for i in layers]
    
    if w is None :
        src_feats = [F.normalize(s_f.squeeze(), p=2, dim=0) for s_f in src_feats]
        trg_feats = [F.normalize(t_f.squeeze(), p=2, dim=0) for t_f in trg_feats]
       
    else:
        s = torch.cat(src_feats,1)
        src_feats =  w(s)
        src_feats = [F.normalize(src_feats, p=2, dim=1).squeeze()]
 
        t = torch.cat(trg_feats,1)
        trg_feats = w(t)
        trg_feats = [F.normalize(trg_feats, p=2, dim=1).squeeze()]
    
    H,W = grid_dim, grid_dim
    grid = np.mgrid[0:W, 0:H].reshape(2,-1).T
    s_H, s_W = src_im.shape[1:]
    t_H, t_W = trg_im.shape[1:]

    src_xratio = W*1./ s_W
    src_yratio = H*1./ s_H

    trg_xratio = t_W / W*1.
    trg_yratio = t_H / H*1.
    
    if bbox is not None:
        x1,y1,x2,y2 = bbox[0]
        x1,x2 = x1/trg_xratio, x2/trg_xratio
        y1,y2 = y1/trg_yratio, y2/trg_yratio
        x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2) 
    trg_kps = []

    for x,y in src_kps.T:
        mi, mj = int(y * src_yratio), int(x * src_xratio)
        mi, mj = min(H-1,mi), min(W-1,mj)
        map_to_vis = torch.zeros(1,H*W).cuda()

        for s_f, t_f  in zip(src_feats,trg_feats):
            t_f = t_f.reshape(-1, H*W)
            
            map_to_vis += F.softmax(torch.matmul(s_f[:,mi,mj].t(), t_f))
        
        map_to_vis = map_to_vis.reshape(H,W)
        if bbox is not None:
            map_to_vis_n = torch.zeros_like(map_to_vis)
            map_to_vis_n[y1:y2,x1:x2] = map_to_vis[y1:y2,x1:x2] 
            map_to_vis = map_to_vis_n 
    
        ti, tj  = grid[torch.argmax(map_to_vis)]
        trg_kps.append((int(tj*trg_xratio), int(ti*trg_yratio)))

    trg_kps = torch.from_numpy(np.array(trg_kps))
    return trg_kps.T

