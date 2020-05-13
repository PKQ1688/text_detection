# -*- coding:utf-8 -*-
# @author :adolf
import random
import torch

from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as transforms


def get_transforms(train):
    transforms_l = list()
    transforms_l.append(ToTensor())
    if train:
        # transforms_l.append(Resize(256))
        transforms_l.append(RandomHorizontalFlip(0.5))
    # transforms_l.append(ToTensor())
    return Compose(transforms_l)


get_transforms2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_transforms_3():
    transforms_l = list()
    transforms_l.append(ToTensor_p())
    # transforms_l.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return Compose_p(transforms_l)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Compose_p(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            # height, width = image.shape[-2:]
            image = image.flip(-1)
            mask = mask.flip(-1)

        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        return image, mask


class ToTensor_p(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image


class ScaleResize(object):
    def __init__(self, shortest_side=600):
        self.shortest_side = shortest_side

    def __call__(self, image, target):
        width = image.size[0]
        height = image.size[1]

        # scale = float(self.shortest_side) / float(min(height, width))
        if height > width:
            scale = float(self.shortest_side) / float(width)
            image = F.resize(image, (self.shortest_side, int(height * scale)), 2)
        else:
            scale = float(self.shortest_side) / float(height)
            image = F.resize(image, (int(width * scale), self.shortest_side), 2)

        h_scale = float(image.size[1]) / float(height)
        w_scale = float(image.size[0]) / float(width)

        scale_gt = []
        # print('2target', target)
        bbox = target["boxes"]
        for box in bbox:
            scale_box = []
            for i in range(len(box)):
                if i % 2 == 0:
                    scale_box.append(int(int(box[i]) * w_scale))
                else:
                    scale_box.append(int(int(box[i]) * h_scale))
            scale_gt.append(scale_box)
        boxes = torch.as_tensor(scale_gt, dtype=torch.float32)
        target["boxes"] = boxes
        return image, target


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        img = F.resize(img, self.size, self.interpolation)
        mask = F.resize(img, self.size, self.interpolation)
        return img, mask
