"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder, _make_feature_extractor


class FeatureExtNet(BaseModel):

    def __init__(self, backbone, use_pretrained=True, path=None):

        super(FeatureExtNet, self).__init__()


        self.model = _make_feature_extractor(backbone, use_pretrained, path=path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: feature
        """
        
        return self.model(x)

class FeatureNet(BaseModel):

    def __init__(self, backbone, features=256, use_pretrained=True, path=None):

        super(FeatureNet, self).__init__()


        self.pretrained, _ = _make_encoder(backbone, features, use_pretrained, path=path)
        self.backbone = backbone
        #if path:
        #   self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        
        if self.backbone =='resnet50_new':
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)
            layer_5 = self.pretrained.layer5(layer_4)

            out = [layer_1,layer_2,layer_3,layer_4,layer_5]
            return out

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        out = [layer_1,layer_2,layer_3,layer_4]
        return out

