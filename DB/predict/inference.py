# -*- coding:utf-8 -*-
# @author :adolf
import os
import sys
import time
import cv2

import torch
from post_processing.seg_detector_representer import SegDetectorRepresenter
from db_model.model import DBModel
from PIL import Image
import numpy as np
from torchvision import transforms as T

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(T, item['type'])(**args)
        tr_list.append(cls)
    tr_list = T.Compose(tr_list)
    return tr_list


def demo_visualize(image_path, output):
    boxes, _ = output
    boxes = boxes[0]
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = original_image.shape
    pred_canvas = original_image.copy().astype(np.uint8)
    pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

    for box in boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(pred_canvas, [box], True, (0, 0, 255), 1)

    return pred_canvas


class OnePredict(object):
    def __init__(self, params):
        self.params = params
        self.model = DBModel(params['model_config'])
        self.post_processing = SegDetectorRepresenter(thresh=params['thresh'],
                                                      box_thresh=params['box_thresh'],
                                                      max_candidates=params['max_candidates'],
                                                      unclip_ratio=params['unclip_ratio'])
        self.model_path = params['model_path']
        if not os.path.exists(self.model_path):
            print("Checkpoint not found: " + self.model_path)
        self.device = torch.device('cpu')
        self.init_torch_tensor()
        self.resume()
        self.model.eval()
        self.transform = []
        for t in params['transform']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

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

    @staticmethod
    def pre_process(self, img):
        return img

    @staticmethod
    def demo_visualize(image_path, output):
        boxes, _ = output
        boxes = boxes[0]
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_shape = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 0, 255), 1)

        return pred_canvas

    def inference(self, img_path, is_resize=False, is_visualize=True):
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if is_resize:
            scale = self.params['short_size'] / min(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}

        with torch.no_grad():
            print(tensor)
            preds = self.model(tensor)
            print(preds)

            outputs = self.post_processing.represent(batch=batch, pred=preds, is_output_polygon=self.params['polygon'])

            print('output', outputs)

        if is_visualize:
            vis_img = self.demo_visualize(img_path, outputs)
            if not os.path.isdir(self.params['result_dir']):
                os.mkdir(self.params['result_dir'])
            cv2.imwrite(os.path.join(self.params['result_dir'], img_path.split('/')[-1].split('.')[0] + '.jpg'),
                        vis_img)

        return outputs


if __name__ == '__main__':
    params = dict()
    params['polygon'] = True
    params['short_size'] = 736
    params['result_dir'] = 'images_result/'
    params['transform'] = [{'type': 'ToTensor', 'args': {}},
                           {'type': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}]
    params['model_config'] = {
        'backbone': 'deformable_resnet18',
        'pretrained': True,
        'segmentation_body': {'type': 'FPN', 'args': {'inner_channels': 256}},
        'segmentation_head': {'type': 'DBHead', 'args': {'out_channels': 2, 'k': 50}}
    }
    params['thresh'] = 0.1
    params['box_thresh'] = 0.1
    params['max_candidates'] = 1000
    params['unclip_ratio'] = 1.5
    params['model_path'] = 'model_save/db_model_0.pth'

    img_predict = OnePredict(params)
    outputs = img_predict.inference(img_path='images/test/test_image_5.jpg')
