# -*- coding:utf-8 -*-
# @author :adolf
import os
from post_process.seg_detector_representer import SegDetectorRepresenter
from model.dbmodel_v2 import dbnet_resnet50_fpn
# from model.dbmodel import DBModel
import torch
import torch.nn as nn
from torchvision import transforms as T

import cv2
import numpy as np


class OnePredict(object):
    def __init__(self, configs, use_model=None):
        self.model = dbnet_resnet50_fpn()
        # self.model = DBModel()
        self.polygon = configs['inference_params']['polygon']

        self.thresh = configs['inference_params']['thresh']
        self.box_thresh = configs['inference_params']['box_thresh']
        self.max_candidates = configs['inference_params']['max_candidates']
        self.unclip_ratio = configs['inference_params']['unclip_ratio']

        self.DateParallel = configs['train_params']['DateParallel']

        self.post_processing = SegDetectorRepresenter(
            thresh=self.thresh,
            box_thresh=self.box_thresh,
            max_candidates=self.max_candidates,
            unclip_ratio=self.unclip_ratio
        )
        self.model_path = configs['root_path']['model_path']
        if use_model is not None:
            self.model_path = use_model
        if not os.path.exists(self.model_path):
            print("Checkpoint not found: " + self.model_path)

        self.device = torch.device("cpu")
        self.init_torch_tensor()
        self.resume()
        self.model.eval()
        self.transform = []
        for t in configs['data']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = self.get_transforms(self.transform)
        self.result_dir = configs['root_path']['result_dir']
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        self.short_size = configs['inference_params']['short_size']

        print(self.device)
    def init_torch_tensor(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def resume(self):
        if self.DateParallel:
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=True)
        self.model.to(self.device)

    @staticmethod
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

    @staticmethod
    def demo_visualize(image_path, output):
        boxes, _ = output
        boxes = boxes[0]
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # original_image = cv2.resize(original_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
        original_shape = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 0, 255), 1)

        return pred_canvas

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.result_dir, result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.polygon:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        # res.write(result + ',' + str(score) + "\n")
                        res.write(result + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.box_thresh:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        # res.write(result + ',' + str(score) + "\n")
                        res.write(result + "\n")

    def binarizer(self, pred):
        return pred > self.thresh

    def my_post_process(self, pred):
        segmentation = pred[0, 0, :, :]
        print('segmentation', segmentation)
        threshold = pred[0:, 1, :, :]
        segmentation = self.binarizer(segmentation)
        print(segmentation.size())
        segmentation = segmentation.cpu().numpy()
        threshold = threshold.cpu().numpy()

        segmentation = (segmentation * 255).astype(np.uint8)
        threshold = (threshold * 255).astype(np.uint8)
        print(threshold.shape)
        # segmentation = segmentation.transpose(1, 2, 0)
        # threshold = threshold.transpose(1, 2, 0)
        print('222', segmentation.shape)
        print(segmentation)
        cv2.imwrite('images_result/mask.jpg', segmentation)
        cv2.imwrite('images_result/thres.jpg', threshold)

    def inference(self, img_path, is_resize=True, is_visualize=True, is_format_output=False, out_path=None):
        img = cv2.imread(img_path)

        # print('1111', img.shape)
        # img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        # print(h, w)
        if is_resize:
            scale = self.short_size / min(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # print('before transform', img)
        tensor = self.transform(img)
        # print('after transform', tensor)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = dict()
        batch['shape'] = [(h, w)]

        batch['filename'] = [img_path]
        batch['image'] = tensor

        with torch.no_grad():
            # print(tensor.size())
            # print('tensor', tensor)
            preds = self.model(tensor)
            # print(111, preds)

            # self.my_post_process(preds)
            outputs = self.post_processing.represent(batch=batch, pred=preds, is_output_polygon=self.polygon)

            # print('output', outputs)

        # is_visualize = False
        if is_visualize:
            vis_img = self.demo_visualize(img_path, outputs)
            if not os.path.isdir(self.result_dir):
                os.mkdir(self.result_dir)
            cv2.imwrite(os.path.join(self.result_dir, img_path.split('/')[-1].split('.')[0] + '.jpg'),
                        vis_img)

        # is_format_output = True
        if is_format_output is True:
            if out_path is not None:
                self.result_dir = out_path
            if not os.path.isdir(self.result_dir):
                os.mkdir(self.result_dir)
            self.format_output(batch, outputs)

        return outputs


if __name__ == '__main__':
    import yaml
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    img_path = "/home/shizai/data2/ocr_data/china_life_test_data/imgs/img_32.jpg"
    # img_path = 'test_imgs/wen.jpg'
    with open('config/db_resnet50.yaml', 'r') as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)
    img_predict = OnePredict(configs=config, use_model='model_use_weights/DB_611_0.21.pth')

    outputs = img_predict.inference(
        img_path=img_path, is_resize=True,
        is_visualize=False, is_format_output=False)

    s1 = time.time()
    for i in range(100):
        outputs = img_predict.inference(
            img_path=img_path, is_resize=True,
            is_visualize=False, is_format_output=False)
    e1 = time.time()
    print('use time', e1 - s1)
