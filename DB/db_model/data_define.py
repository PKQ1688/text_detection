# -*- coding:utf-8 -*-
# @author :adolf
import os
import sys

import numpy as np

import torch
from PIL import Image

import torch.utils.data as data
import detection_util.utils as utils


class CurrentOcrData(object):
    def __init__(self, root, transforms=None, target_transforms=None):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        self.gts = list(sorted(os.listdir(os.path.join(root, "gts"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, 'imgs', self.imgs[item])
        gt_path = os.path.join(self.root, 'gts', self.gts[item])

        img = Image.open(img_path).convert("RGB")
        gt = self.read_text_gt(gt_path)

        targets = {}

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

    def get_annotation(self, gt_path):
        boxes = list()
        texts = list()
        difficult = list()
        with open(gt_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                print(params)


if __name__ == '__main__':
    dataset = CurrentOcrData('/home/shizai/data2/ocr_data/rctw')
    dataset.get_annotation('/home/shizai/data2/ocr_data/rctw/gt/rctw_image_3629.txt')
