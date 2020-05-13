# -*- coding:utf-8 -*-
# @author :adolf
import torch
from transforms import get_transforms_3

from unet_model import UNet
import cv2
from PIL import Image, ImageOps
import math

import numpy as np

import time


# from seg_detector_representer import SegDetectorRepresenter


class OnePredict(object):
    def __init__(self, params):
        self.params = params

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_path = params['model_path']

        self.model = UNet(in_channels=3, out_channels=1)

        self.threshold = 0.5

        self.resume()
        # self.model.eval()

        self.transform = get_transforms_3()

        self.is_resize = True
        self.image_short_side = 1024
        self.init_torch_tensor()
        self.model.eval()

    def init_torch_tensor(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
        # self.model.to(self.device)

    def resume(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.model.to(self.device)

    def resize_img(self, img):
        '''输入PIL格式的图片'''
        width, height = img.size
        # print('111', img.size)
        if self.is_resize:
            if height < width:
                new_height = self.image_short_side
                new_width = int(math.ceil(new_height / height * width / 32) * 32)
            else:
                new_width = self.image_short_side
                new_height = int(math.ceil(new_width / width * height / 32) * 32)
        else:
            if height < width:
                scale = int(height / 32)
                new_image_short_side = scale * 32
                new_height = new_image_short_side
                new_width = int(math.ceil(new_height / height * width / 32) * 32)
            else:
                scale = int(width / 32)
                new_image_short_side = scale * 32
                new_width = new_image_short_side
                new_height = int(math.ceil(new_width / width * height / 32) * 32)
        # print('test1:', np.array(img))
        # print('new:', (new_width, new_height))
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        # print(new_height, new_width)
        # print('test2:', np.array(resized_img))
        return resized_img

    def format_output(self):
        pass

    @staticmethod
    def pre_process(img):
        return img

    @staticmethod
    def pad_sample(img):
        a = img.size[0]
        b = img.size[1]
        if a == b:
            return img
        diff = (max(a, b) - min(a, b)) / 2.0
        if a > b:
            padding = (0, int(np.floor(diff)), 0, int(np.ceil(diff)))
        else:
            padding = (int(np.floor(diff)), 0, int(np.ceil(diff)), 0)

        img = ImageOps.expand(img, border=padding, fill=0)  ##left,top,right,bottom

        assert img.size[0] == img.size[1]
        return img

    def post_process(self, preds, img):
        mask = preds > self.threshold
        mask = mask * 255
        # print(mask.size())
        mask = mask.cpu().numpy()[0][0]
        # print(mask)
        # print(mask.shape())
        cv2.imwrite('mask.png', mask)

        mask = np.array(mask, np.uint8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)

        # img = img.cpu()
        img = np.array(img, np.uint8)

        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        cv2.imwrite('result2.png', img)
        boxes = []

        return boxes

    @staticmethod
    def demo_visualize():
        pass

    def inference(self, img_path, is_visualize=True, is_format_output=False):
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert("RGB")
        # img = Image.open(img_path).convert("RGB")
        # print('222', np.array(img))
        # img = self.pad_sample(img)
        img = self.resize_img(img)
        # print('333', img.size)
        # print('-----', np.array(img))
        ori_img = img
        img.save('img.png')
        # img = [img]
        print('111', np.array(img))
        img = self.transform(img)
        print('222', np.array(img))
        img = img.unsqueeze(0)
        img = img.to(self.device)
        # print('1111', img.size())
        # print(img)

        # print(img)
        with torch.no_grad():
            s1 = time.time()
            preds = self.model(img)
            print(preds)
            s2 = time.time()
            print(s2 - s1)
            # boxes, scores = SegDetectorRepresenter().represent(pred=preds, height=h, width=w, is_output_polygon=False)
            boxes = self.post_process(preds, ori_img)

        # print(preds.shape)

        # is_format_output = True
        # if is_format_output is True:
        #     if not os.path.isdir(self.params['result_dir']):
        #         os.mkdir(self.params['result_dir'])
        #     self.format_output()

        # is_visualize = False
        # if is_visualize:
        #     vis_img = self.demo_visualize()
        #     if not os.path.isdir(self.params['result_dir']):
        #         os.mkdir(self.params['result_dir'])
        #     cv2.imwrite(os.path.join(self.params['result_dir'], img_path.split('/')[-1].split('.')[0] + '.jpg'),
        #                 vis_img)

        # return boxes, scores


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    a = {'model_path': './weights/unet_xia_adam.pth'}
    # OnePredict(a).inference("/home/shizai/data2/ocr_data/midv_500/imgs/KA11_24.jpg")
    # OnePredict(a).inference("/home/shizai/data2/ocr_data/idcard/img/w_24.jpg")
    OnePredict(a).inference("/home/shizai/data2/ocr_data/china_life_test_data/imgs/img_33.jpg")
