# -*- coding:utf-8 -*-
# @author :adolf
import torch.nn as nn
from model.segmentation_db import DBHead
from backbone.resnet import *
from model.segmentation_basic import FPN

import torch.nn.functional as F
import torch


class DBModel(nn.Module):
    def __init__(self, pretrained=False):
        """
        DBNET
        :param model_config: 模型配置
        """
        super().__init__()

        self.backbone = resnet50(pretrained=pretrained)
        self.segmentation_body = FPN(backbone_out_channels=[256, 512, 1024, 2048])
        self.segmentation_head = DBHead()

    def forward(self, x):
        # print(model_config['segmentation_body']['args'])
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        # print(111, segmentation_body_out.size())
        y = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    device = torch.device("cpu")

    x = torch.zeros(1, 3, 640, 640).to(device)

    model = DBModel().to(device)
    import time

    print('222')
    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model)
