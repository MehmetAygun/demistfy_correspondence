import torch.nn.functional as F
import time
import torch
import tps


def sup_loss(src_f, trg_f, src_kps, trg_kps):
    
    B, C, H, W = src_f.shape
    device = src_f.device
    xi = torch.linspace(0, W - 1, W)
    yi = torch.linspace(0, H - 1, H)
    yy, xx = torch.meshgrid(yi, xi)
    xxyy = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    xxyy = xxyy.reshape(H, W, 2)
    xxyy = xxyy.to(device)  
    loss = 0. 
    
    src_f = F.normalize(src_f, p=2, dim=1) * 5
    trg_f = F.normalize(trg_f, p=2, dim=1) * 5
    
    n = 0
    b_corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W) 
    sb_corr = F.softmax(b_corr.view(B,H,W,-1), dim=3).view(b_corr.shape)
    valid_kps = ((src_kps[:,:,0] > 0) & (trg_kps[:,:,0] > 0))  
    for b in range(B):
        s = src_kps[b,valid_kps[b]].long()
        t = trg_kps[b,valid_kps[b]].long()
        diff = xxyy[:,:,None,:] - t[None,None,:]
        diff = torch.sum(torch.pow(torch.pow(diff,2),0.5),3)
        l = torch.sum(sb_corr[b,s[:,1],s[:,0]].permute(1,2,0) * diff)
        loss = loss + l
    return loss / (H*W)



def asym_ce_loss(src_f, trg_f, src_pf, trg_pf, t1=10, t2=5):
    
    src_f = F.normalize(src_f, p=2, dim=1) * t1
    trg_f = F.normalize(trg_f, p=2, dim=1) * t1
    
    src_pf = F.normalize(src_pf, p=2, dim=1) * t2
    trg_pf = F.normalize(trg_pf, p=2, dim=1) * t2
    
    B, C, H, W = src_f.shape
    _, Cp, _, _ = src_pf.shape
     
    corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W)
    s_corr = F.softmax(corr.view(B,H,W,-1), dim=3).view(B,H,W,H,W)

    corr_p = torch.bmm(src_pf.view(B,Cp,H*W).permute(0,2,1), trg_pf.view(B,Cp,H*W)).view(B,H,W,H,W)
    s_corr_p = F.softmax(corr_p.view(B,H,W,-1), dim=3).view(B,H,W,H,W)
    
    loss  = torch.sum((-1.*s_corr) * torch.log(s_corr_p+1e-8))/ (B*H*W)
    return loss#CE loss 


def asym_loss(src_f, trg_f, src_pf, trg_pf, t1=10, t2=5):
    
    src_f = F.normalize(src_f, p=2, dim=1) * t1
    trg_f = F.normalize(trg_f, p=2, dim=1) * t1
    
    src_pf = F.normalize(src_pf, p=2, dim=1) * t2
    trg_pf = F.normalize(trg_pf, p=2, dim=1) * t2
    
    r = t2/(t1*1.0)
    B, C, H, W = src_f.shape
    _, Cp, _, _ = src_pf.shape
     
    corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W)
    s_corr = F.softmax(corr.view(B,H,W,-1), dim=3).view(B,H,W,H,W)

    corr_p = torch.bmm(src_pf.view(B,Cp,H*W).permute(0,2,1), trg_pf.view(B,Cp,H*W)).view(B,H,W,H,W)
    s_corr_p = F.softmax(corr_p.view(B,H,W,-1), dim=3).view(B,H,W,H,W)
    
    loss = torch.sum(torch.pow(s_corr - s_corr_p, 2)) 
    return loss*100 / (H*W*B)


def lead_mse_loss(src_f, trg_f, src_pf, trg_pf, t1):
    
    src_f = F.normalize(src_f, p=2, dim=1)
    trg_f = F.normalize(trg_f, p=2, dim=1)
    
    src_pf = F.normalize(src_pf, p=2, dim=1)
    trg_pf = F.normalize(trg_pf, p=2, dim=1)
    
    B, C, H, W = src_f.shape
    _, Cp, _, _ = src_pf.shape
     
    corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W)
    s_corr = F.softmax(corr.view(B,H,W,-1)*t1, dim=3).view(B,H,W,H,W)

    corr_p = torch.bmm(src_pf.view(B,Cp,H*W).permute(0,2,1), trg_pf.view(B,Cp,H*W)).view(B,H,W,H,W)
    s_corr_p = F.softmax(corr_p.view(B,H,W,-1)*t1, dim=3).view(B,H,W,H,W)
    
    loss = torch.sum(torch.pow(s_corr - s_corr_p, 2)) / (B*H*W)

    return loss * 100


def lead_loss(src_f, trg_f, src_pf, trg_pf, t1):
    
    #LEAD loss, Karmali et.al 2022
    src_f = F.normalize(src_f, p=2, dim=1)
    trg_f = F.normalize(trg_f, p=2, dim=1)
    
    src_pf = F.normalize(src_pf, p=2, dim=1)
    trg_pf = F.normalize(trg_pf, p=2, dim=1)
    
    B, C, H, W = src_f.shape
    _, Cp, _, _ = src_pf.shape
     
    corr = torch.bmm(src_f.view(B,C,H*W).permute(0,2,1), trg_f.view(B,C,H*W)).view(B,H,W,H,W)
    s_corr = F.softmax(corr.view(B,H,W,-1)*t1, dim=3).view(B,H,W,H,W)

    corr_p = torch.bmm(src_pf.view(B,Cp,H*W).permute(0,2,1), trg_pf.view(B,Cp,H*W)).view(B,H,W,H,W)
    s_corr_p = F.softmax(corr_p.view(B,H,W,-1)*t1, dim=3).view(B,H,W,H,W)
    
    loss  = torch.sum((-1.*s_corr) * torch.log(s_corr_p+1e-8))/ (B*H*W)

    return loss


def cl_loss(feat, input_size, temperature, feat_spectral, pow=0.5, normalize_vectors=True):
    
    # CL loss, Cheng et al 2021 
    B, C, H, W = input_size
    b, c, h, w = feat.size()
    device = feat.device
    stride = H // h
    if H%h != 0:
        H, W = h * stride, w* stride
    xi = torch.linspace(0, W - 1, W)
    yi = torch.linspace(0, H - 1, H)
    yy, xx = torch.meshgrid(yi, xi)
    yyxx = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    yyxx =  yyxx.reshape(H, W, 2).to(device)

    diff = yyxx[::stride, ::stride, None, None, :] - yyxx[None, None, ::stride, ::stride, :]
    diff = (diff * diff).sum(4).sqrt()
    diff = diff.pow(pow)
    f1 = F.normalize(feat,p=2,dim=1)
    corr = torch.bmm(f1.view(b,c,h*w).permute(0,2,1), f1.view(b,c,h*w))
    smcorr = F.softmax(corr.reshape(b, h, w, h*w) * temperature, dim=3).reshape(b,h,w,h,w)
    L = diff.repeat(b,1,1,1,1) * smcorr
    loss = L.sum()
    
    return loss / (h * w * b)


def dense_correlation_loss(feats, meta, pow=0.5, fold_corr=False, normalize_vectors=True):
    
    device = feats.device
    grid = meta['grid']

    # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    grid = grid.to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]
    
    feats1 = feats[0::2]
    feats2 = feats[1::2]
    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]
    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)

    if fold_corr:
        from folded_correlation import DenseCorr
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        assert not normalize_vectors
        dense_corr = DenseCorr.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)

    
    if normalize_vectors:
        feats1 = F.normalize(feats1, p=2, dim=1) * 20
        feats2 = F.normalize(feats2, p=2, dim=1) * 20
    corr = torch.bmm(feats1.view(B, C, H*W).permute(0,2,1), feats2.view(B,C,H*W))
    corr = corr.reshape(B, H, W, h, w)
    smcorr = F.softmax(corr.reshape(B, H, W, -1), dim=3).reshape(corr.shape)
   
    xxyy = xxyy.repeat(B,1,1,1)
    diff = batch_grid_u[:, :, :, None, None, :] - \
            xxyy[:, None, None, ::stride, ::stride, :]
    diff = (diff * diff).sum(5).sqrt()
    diff = diff.pow(pow)
    L = smcorr * diff
    loss = L.sum()
    return loss / (H * W * B)
    
    """
    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target

        if normalize_vectors:
            f1 = F.normalize(f1, p=2, dim=0) * 20
            f2 = F.normalize(f2, p=2, dim=0) * 20

        corr = torch.matmul(f1.t(), f2)
        corr = corr.reshape(H, W, h, w)

        with torch.no_grad():
            diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        # grid_u = tps.grid_unnormalize(grid[b], H_input, W_input)
        # diff = grid_u[:, :, None, None, :] - xxyy[None, None, :, :, :]

        # Equivalent to this
        #
        # diff = torch.zeros(H_input, W_input, H_input, W_input, 2)
        # for I in range(H_input):
        #     for J in range(W_input):
        #         for i in range(H_input):
        #             for j in range(W_input):
        #                 diff[I, J, i, j, 0] = J + flow[b, I, J, 0] - j
        #                 diff[I, J, i, j, 1] = I + flow[b, I, J, 1] - i

        # diff = diff[::stride, ::stride, ::stride, ::stride]
        # diff = (diff * diff).sum(4).sqrt()
        # diff = diff.pow(pow)

        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)

        L = diff * smcorr

        loss += L.sum()
    return loss / (H * W * B)
    """

def estimate_mem(x):
    if x.dtype == torch.float32:
        nbytes = 4
    elif x.dtype == torch.float16:
        nbytes = 2
    elif x.dtype == torch.int8:
        nbytes = 1
    else:
        import ipdb; ipdb.set_trace()
    return torch.numel(x) * nbytes / (1024) ** 3


def dense_correlation_loss_dve(feats, meta, pow=0.5, fold_corr=False, normalize_vectors=True):
    feats = feats
    device = feats.device

    # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    grid = meta['grid'].to(device)
    
    t = meta['t']
    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]

    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)
    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]

    if False:
        import matplotlib.pyplot as plt

        vis1 = meta['im1'][0].clone()
        vis2 = meta['im2'][0].clone()
        visgrid = tps.grid_unnormalize(grid, H_input, W_input)[0]

        fig = plt.figure()  # a new figure window
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.imshow(vis1.permute(1,2,0)+0.5)
        ax2.imshow(vis2.permute(1,2,0)+0.5)

        for i in range(H_input):
            for j in range(W_input):
                if torch.rand([]) < 0.01:
                    ax1.scatter(j,i)
                    jj,ii = visgrid[i,j]
                    ax2.scatter(jj,ii)

        dists = (batch_grid_u[0] - xxyy[::stride,::stride]).pow(2).sum(2).sqrt()
        ax3.imshow(dists/dists.max())
        fig.savefig('/tmp/lossvis.pdf')
        fig.clf()

    if fold_corr:
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        from folded_correlation_dve import DenseCorrDve
        dense_corr = DenseCorrDve.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride,
                          normalize_vectors, pow)

    aux_idxs = [(b + B//2) % B for b in range(B)]
    if normalize_vectors:
        feats1 = F.normalize(feats1, p=2, dim=1) * t
        feats2 = F.normalize(feats2, p=2, dim=1) * t
        featsa = F.normalize(feats1[aux_idxs], p=2, dim=1) * t
    
    corr = torch.bmm(feats1.view(B, C, H*W).permute(0,2,1), featsa.view(B,C,H*W))
    corr = corr.reshape(B, H, W, h, w)
    smcorr = F.softmax(corr.reshape(B, H, W, -1), dim=3).reshape(corr.shape)

    smcorr_fa = smcorr[:,None, ...] * featsa.reshape(B, -1, 1, 1, h, w)
    f1_via_fa = smcorr_fa.sum((4, 5)).reshape(B, C, H * W)

    corr2 = torch.bmm(f1_via_fa.permute(0,2,1), feats2.view(B, C, H*W))
    smcorr2 = F.softmax(corr2.reshape(B, H, W, -1), dim=3).reshape(corr.shape)
    
    xxyyb = xxyy.repeat(B,1,1,1)
    diff = batch_grid_u[:, :, :, None, None, :] - \
            xxyyb[:, None, None, ::stride, ::stride, :]
    diff = (diff * diff).sum(5).sqrt()
    diff = diff.pow(pow)
    
    L = diff * smcorr2

    loss = L.float().sum()
    return loss / (H*W*B)
    """
    loss = 0. 
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target
        fa = feats1[(b + B//2) % B].reshape(C, h * w)  # auxiliary

        corr = torch.matmul(f1.t(), fa)
        corr = corr.reshape(H, W, h, w)
        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)
        smcorr_fa = smcorr[None, ...] * fa.reshape(-1, 1, 1, h, w)
        del smcorr

        f1_via_fa = smcorr_fa.sum((3, 4)).reshape(C, H * W)
        del smcorr_fa

        corr2 = torch.matmul(f1_via_fa.t(), f2).reshape(corr.shape)
        smcorr2 = F.softmax(corr2.reshape(H, W, -1), dim=2).reshape(corr.shape)
        del corr2

        with torch.no_grad():
            diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        L = diff * smcorr2

        loss += L.float().sum()
    print (loss, loss1)
    return loss / (H * W * B)
    """

def dense_correlation_loss_trick(feats, meta, pow=0.5, fold_corr=False,
        normalize_vectors=True):
    feats = feats[0]
    device = feats.device
    grid = meta['grid']

    # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    grid = grid.to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]

    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]
    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)

    if fold_corr:
        from folded_correlation import DenseCorr
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        assert not normalize_vectors
        dense_corr = DenseCorr.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target

        if normalize_vectors:
            f1 = F.normalize(f1, p=2, dim=0) * 20
            f2 = F.normalize(f2, p=2, dim=0) * 20

        corr = torch.matmul(f1.t(), f2)
        corr = corr.reshape(H, W, h, w)

        with torch.no_grad():
            # replace with expanded terms for efficiency
            import ipdb; ipdb.set_trace()
            diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        # grid_u = tps.grid_unnormalize(grid[b], H_input, W_input)
        # diff = grid_u[:, :, None, None, :] - xxyy[None, None, :, :, :]

        # Equivalent to this
        #
        # diff = torch.zeros(H_input, W_input, H_input, W_input, 2)
        # for I in range(H_input):
        #     for J in range(W_input):
        #         for i in range(H_input):
        #             for j in range(W_input):
        #                 diff[I, J, i, j, 0] = J + flow[b, I, J, 0] - j
        #                 diff[I, J, i, j, 1] = I + flow[b, I, J, 1] - i

        # diff = diff[::stride, ::stride, ::stride, ::stride]
        # diff = (diff * diff).sum(4).sqrt()
        # diff = diff.pow(pow)

        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)

        L = diff * smcorr

        loss += L.sum()

    return loss / (H * W * B)


def rel_diff(x1, x2, name):
    out = torch.abs(x1 - x2).sum() / torch.abs(x2).mean()
    print("rel diff for {}: {}".format(name, out))


def dense_corr_trick_check():
    dve_dim = 4
    B, C, H, W = 4, dve_dim, 4, 4

    common = {"dtype": torch.double, "requires_grad": True}
    feats = torch.randn(B, C, H, W, **common)
    batch_grid_u = torch.randn(B, H, W, 2, dtype=torch.double,
                               requires_grad=False)

    feats = feats.cuda().float()
    batch_grid_u = batch_grid_u.cuda().float()
    out = dense_correlation_loss([feats], {"grid": batch_grid_u})
    out2 = dense_correlation_loss_trick([feats], {"grid": batch_grid_u})
    rel_diff(out, out2, "trick")


