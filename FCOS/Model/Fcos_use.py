# -*- coding:utf-8 -*-
# @author :adolf
from torch import nn
from Model.FCOS import FCOS
from Model.layers import ShapeSpec
from Model.Backbone import build_fcos_resnet_fpn_backbone
# from Model.structures import Instances

from addict import Dict
import yaml


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        self.backbone = build_fcos_resnet_fpn_backbone(cfg, input_shape)
        self.output_shape = self.backbone.output_shape()
        # print(self.output_shape)
        self.segmentation_head = FCOS(cfg, self.output_shape)
        self.top_module = self.build_top_module(cfg)

    def forward(self, x, images, gt=None):
        features = self.backbone(x)
        print(features.keys())
        results, losses = self.segmentation_head(images, features,
                                                 gt_instances=gt,
                                                 top_module=self.top_module)
        return results, losses

    @staticmethod
    def build_top_module(cfg):
        top_type = cfg.TOP_MODULE.NAME
        if top_type == "conv":
            inp = cfg.FPN.OUT_CHANNELS
            oup = cfg.TOP_MODULE.DIM
            top_module = nn.Conv2d(
                inp, oup,
                kernel_size=3, stride=1, padding=1)
        else:
            top_module = None
        return top_module


if __name__ == '__main__':
    import torch

    config_file = "config.yaml"
    with open(config_file, encoding="utf-8", mode="r") as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)
    cfg = Dict(config)

    x = torch.zeros(1, 3, 640, 640).to("cpu")
    # x = [x]
    images = Dict()
    images.image_sizes = [(640, 640)]
    # print(type(images.image_sizes))
    gt = None
    '''
    gt = [Instances(num_instances=8, image_height=1333, image_width=750, fields=
        [gt_boxes: Boxes(
        tensor([[566., 749., 674., 774.],
                [426., 751., 516., 775.],
                [364., 802., 674., 829.],
                [310., 862., 677., 887.],
                [310., 898., 594., 927.],
                [456., 937., 595., 962.],
                [520., 1005., 678., 1032.],
                [98., 1006., 484., 1036.]], device='cuda:0')), gt_classes: tensor([0, 0, 0, 0, 0, 0, 0, 0],
                                                                                  device='cuda:0')])]
    '''
    fcos_model = MyModel(cfg)
    print(fcos_model)

    fcos_model.eval()
    results, losses = fcos_model(x, gt, images)
    print(results)
    print(losses)
