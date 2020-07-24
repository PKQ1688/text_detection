# -*- coding:utf-8 -*-
# @author :adolf
from Model.Backbone import build_resnet_fpn_backbone
from Model.layers import ShapeSpec
import yaml
from addict import Dict

config_file = "config.yaml"
with open(config_file, 'r') as fp:
    config = yaml.load(fp.read(), Loader=yaml.FullLoader)

print(config)
cfg = Dict(config)

input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
backbone = build_resnet_fpn_backbone(cfg, input_shape)
print(backbone.output_shape())

