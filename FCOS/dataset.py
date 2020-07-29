# -*- coding:utf-8 -*-
# @author :adolf
import os
import cv2
from Model.structures import Instances, Boxes

from typing import Tuple, List

from tqdm import tqdm
import torch
from torchvision import transforms as T

import time
import numpy as np
from addict import Dict

import random


class TextUseData:
    def __init__(self, root, is_training=True, transforms=None):
        self.is_training = is_training
        self.root = root
        self.transforms = transforms

        self.imgs_list = list(sorted(os.listdir(os.path.join(root, "imgs"))))

        random.shuffle(self.imgs_list)

        validation_cases = int(0.1 * len(self.imgs_list))
        #
        validation_patients = random.sample(self.imgs_list, k=validation_cases)
        # validation_patients = ['img_363.jpg']

        if not is_training:
            self.imgs_list = validation_patients
        else:
            self.imgs_list = sorted(
                list(set(self.imgs_list).difference(validation_patients))
            )

    def __len__(self):
        return len(self.imgs_list)

    @staticmethod
    def order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def transforms_train_data(img):
        return img

    @staticmethod
    def transforms_eval_data(img):
        return img

    def get_annotation(self, gt_path):
        polys = list()
        texts = list()
        ignores = list()
        gt_boxes = list()
        gt_classes = list()
        with open(gt_path, encoding="utf-8", mode="r") as fp:
            for line in fp.readlines():
                params = line.strip().strip('ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = self.order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        polys.append(box)
                        texts.append(params[8])
                        ignores.append(params[8] in ['*', '###'])
                        x1 = min(box[0][0], box[3][0])
                        x2 = max(box[1][0], box[2][0])
                        y1 = min(box[0][1], box[1][1])
                        y2 = max(box[2][1], box[3][1])
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(0)
                except Exception as e:
                    print(e)
                    print('get annotation is failed {}'.format(gt_path))

        data = Dict()
        data.text_poly = np.array(polys)
        data.texts = texts
        data.ignore_tags = ignores
        data.gt_boxes = np.array(gt_boxes)
        data.gt_classes = np.array(gt_classes)

        return data

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "imgs", self.imgs_list[index])
        gt_name = self.imgs_list[index].replace("jpg", "txt")

        gt_path = os.path.join(self.root, 'gts', gt_name)

        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_training:
            img_tensor = self.transforms_train_data(img)
            data = self.get_annotation(gt_path)
            return img_tensor, data

        else:
            img_tensor = self.transforms_eval_data(img)
            return img_tensor, {}
