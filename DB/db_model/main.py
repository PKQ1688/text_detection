# -*- coding:utf-8 -*-
# @author :adolf
import sys
import torch
import detection_util.transforms as T

from detection_util.engine import train_one_epoch
import detection_util.utils as utils

from db_model.data_define import *

import torch.utils.data as data

from db_model.model import DBModel
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_transform(train):
    transforms = list()
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 2

    # dataset =

    model_config = {
        'backbone': 'resnet18',
        'pretrained': True,
        'segmentation_body': {'type': 'FPN', 'args': {'inner_channels': 256}},
        'segmentation_head': {'type': 'DBHead', 'args': {'out_channels': 2, 'k': 50}}
    }
    model = DBModel(model_config=model_config).to(device)


