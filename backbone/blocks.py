import torch
import torch.nn as nn

import torchvision.models as models

from .vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    forward_vit,
)

import pickle

from .convnext import (
   convnext_small, convnext_tiny, convnext_base 
)

def _make_feature_extractor(backbone, use_pretrained, path=None):
      
    if backbone == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
        if path:
            print ('loading {}'.format(path))
            checkpoint = torch.load(path)
            dict_new = {}
            for k,v in checkpoint['state_dict'].items():
                if 'base_encoder' in k:
                    n_k = k.replace('module.base_encoder.','')
                    dict_new[n_k] = v
            if len(dict_new.keys()) == 0:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(dict_new,strict=False)
        model.fc = torch.nn.Identity()
        
    elif backbone == "inat_resnet50":
        model = models.resnet50(pretrained=False)
    
        if 'inat2021_supervised' in path: 
            model.fc = torch.nn.Linear(model.fc.in_features, 10000)
            checkpoint = torch.load(path, map_location="cpu")
            msg = model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.fc = nn.Identity()
        elif 'mocov2' in path:
            model.fc = torch.nn.Identity()
            checkpoint = torch.load(path, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=True)
        else:
            assert False

    elif backbone == "convnext":
        model = convnext_base(pretrained=use_pretrained)
        model.head = nn.Identity()

    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
    
    return model


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None, use_vit_only=False, use_readout="ignore", path=None):
    
    if backbone == "vitl16_384":
        pretrained = _make_pretrained_vitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # ViT-L/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb_rn50_384":
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)
    elif backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)     # efficientnet_lite3  
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3     
    
    elif backbone == "resnet50" or backbone == "resnet50_new":
        pretrained = _make_pretrained_resnet(use_pretrained, backbone, path)
        scratch = None
    elif backbone == "resnet101":
        pretrained = _make_pretrained_resnet(use_pretrained, backbone)
        scratch = None  
    
    elif backbone == "inat_resnet50":
        pretrained = _make_pretrained_inat_resnet(path)
        scratch = None
    
    elif backbone == "coco_resnet50":
        pretrained = _make_pretrained_coco_resnet(path)
        scratch = None
    
    elif backbone == "convnext":
        pretrained = _make_pretrained_convnext(use_pretrained)
        scratch = None
    
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand==True:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained
    

def _make_convnext_backbone(convnext):
    
    pretrained = nn.Module()
    
    pretrained.layer1 = nn.Sequential(*convnext.downsample_layers[0], *convnext.stages[0])
    pretrained.layer2 = nn.Sequential(*convnext.downsample_layers[1], *convnext.stages[1])
    
    pretrained.layer3 = nn.Sequential(*convnext.downsample_layers[2], *convnext.stages[2][:10])
    pretrained.layer4 = nn.Sequential(convnext.stages[2][10:],*convnext.downsample_layers[3], *convnext.stages[3])
    #pretrained.layer3 = nn.Sequential(*convnext.downsample_layers[2], *convnext.stages[2])
    #pretrained.layer4 = nn.Sequential(*convnext.downsample_layers[3], *convnext.stages[3])
 
   
    return pretrained

def _make_resnet_feature_extractor(resnet):
    
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_resnet_backbone_new(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
    )
    
    pretrained.layer2 = resnet.layer1
    pretrained.layer3 = resnet.layer2
    pretrained.layer4 = resnet.layer3
    pretrained.layer5 = resnet.layer4

    return pretrained


def _make_pretrained_convnext(use_pretrained):
    convnext = convnext_base(pretrained=use_pretrained)
    return _make_convnext_backbone(convnext)


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


def _make_pretrained_resnet(use_pretrained, backbone, path=None):
    if backbone == 'resnet50' or backbone =='resnet50_new':
        resnet = models.resnet50(pretrained=use_pretrained)
        if path:
            print ('loading {}'.format(path))
            checkpoint = torch.load(path)
            dict_new = {}
            for k,v in checkpoint['state_dict'].items():
                if 'base_encoder' in k:
                    n_k = k.replace('module.base_encoder.','')
                    dict_new[n_k] = v
            
            if len(dict_new.keys()) == 0:
                msg = resnet.load_state_dict(checkpoint['state_dict'], strict=False)
                print (msg)
            else:
                msg = resnet.load_state_dict(dict_new,strict=False)
                print (msg)
    elif backbone == 'resnet101':
        resnet = models.resnet101(pretrained=use_pretrained)
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
    
    if backbone == 'resnet50_new':
        return _make_resnet_backbone_new(resnet)
    return _make_resnet_backbone(resnet)


def _make_pretrained_coco_resnet(path):
    
    model = models.resnet50(pretrained=False)
    with open(path,'rb') as f:
        d = pickle.load(f)
    newmodel = {}
    for k in list(d['model'].keys()):
        old_k = k
        if not "backbone.bottom_up" in k:
            continue
        k = k.replace("backbone.bottom_up.","")
        k = k.replace("shortcut.norm", "downsample.1")
        k = k.replace("shortcut","downsample.0")
        for t in [1, 2, 3]:
            k = k.replace("conv{}.norm".format(t),"bn{}".format(t))
        for t in [1, 2, 3, 4]:
            k = k.replace("res{}".format(t + 1),"layer{}".format(t))  
        if "layer" not in k:
            k = k.replace("stem.",'')
        newmodel[k] = torch.from_numpy(d['model'][old_k])
    
    msg = model.load_state_dict(newmodel, strict=False)
    return _make_resnet_backbone(model)


def _make_pretrained_inat_resnet(path):
    
    model = models.resnet50(pretrained=False)
    
    if 'inat2021_supervised' in path: 
        model.fc = torch.nn.Linear(model.fc.in_features, 10000)
        checkpoint = torch.load(path, map_location="cpu")
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)

    elif 'mocov2' in path:
        model.fc = torch.nn.Identity()
        checkpoint = torch.load(path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)

    elif 'swav_mini' in path:
        model.fc = torch.nn.Identity()
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}

        for k in list(state_dict.keys()):
            if 'projection' in k or 'prototypes' in k:
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)
    
    elif 'aves' in path:
        model.fc = torch.nn.Linear(model.fc.in_features, 128) 
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = {k.replace("encoder.module.", ""): v for k, v in checkpoint['model'].items()}
        msg = model.load_state_dict(state_dict, strict=False)

    else:
        print("Wrong path for inat {} ".format(path))
        assert False
    print (msg)
    return _make_resnet_backbone(model)



class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output




class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

