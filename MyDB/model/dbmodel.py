# -*- coding:utf-8 -*-
# @author :adolf
import torch.nn.functional as F
import torch.nn as nn
import torch

from backbone.backbone_utils import resnet_fpn_backbone
from model.segmentation_db import DBHead


class DBModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        """
        DBNET
        :param model_config: 模型配置
        """
        super().__init__()
        self.backbone = backbone
        self.out_channels = self.backbone.out_channels
        self.segmentation_head = DBHead(in_channels=256)

    def forward(self, x):
        # print(model_config['segmentation_body']['args'])
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        print(111, backbone_out.size())
        y = self.segmentation_head(backbone_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


def dbnet_resnet50_fpn(pretrained=False):
    backbone = resnet_fpn_backbone('resnet', pretrained)
    db_model = DBModel(backbone)
    return db_model


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x = torch.zeros(1, 3, 640, 640).to(device)

    model = dbnet_resnet50_fpn().to(device)
    import time

    print('222')
    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
