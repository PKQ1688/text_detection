# -*- coding:utf-8 -*-
# @author :adolf
from torch import nn
from Model.FCOS import FCOS
from Model.layers import ShapeSpec
from Model.Backbone import build_fcos_resnet_fpn_backbone

from addict import Dict
import yaml

class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        self.backbone = build_fcos_resnet_fpn_backbone(cfg, input_shape)
        self.output_shape = self.backbone.output_shape()
        print(self.output_shape)
        self.segmentation_head = FCOS(cfg, self.output_shape)

    def forward(self, x):
        features = self.backbone(x)
        y = self.segmentation_head(features)
        return y


if __name__ == '__main__':
    config_file = "config.yaml"
    with open(config_file, 'r') as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)
    cfg = Dict(config)

    fcos_model = MyModel(cfg)
    print(fcos_model)
