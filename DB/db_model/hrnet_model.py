# -*- coding:utf-8 -*-
# @author :adolf
import torch.nn.functional as F
import torch.nn as nn
import torch

from backbone.HRNet.seg_hrnet import get_seg_model
from backbone.segmentation_head import DBHead
import yaml


class HRNetDBModel(nn.Module):
    def __init__(self, db_args):
        super().__init__()
        with open('base_model/HRNet/seg_test.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # print(config)
        self.backbone = get_seg_model(config)
        self.segmentation_head = DBHead(720, **db_args)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        y = self.segmentation_head(backbone_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    db_args = {'out_channels': 2,
               'k': 50}
    model = HRNetDBModel(db_args)
    x = torch.zeros(1, 3, 640, 640)
    y = model(x)
    print(model)
    print('3333',y.size())
