# -*- coding:utf-8 -*-
# @author :adolf
import os

from data_util import *
import cv2


class CurrentOcrData(object):
    def __init__(self, root, pre_processes=None, transforms=None, filter_keys=None, ignore_tags=None, is_training=True):
        self.is_training = is_training
        self.root = root
        self.transforms = transforms
        self.pre_processes = pre_processes
        self.filter_key = filter_keys
        self.ignore_tags = ignore_tags

        self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        self.gts = list(sorted(os.listdir(os.path.join(root, "gts"))))
        self.init_pre_process()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, 'imgs', self.imgs[item])
        gt_path = os.path.join(self.root, 'gts', self.gts[item])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        targets = self.get_annotation(gt_path)
        targets['img'] = img
        targets['shape'] = [img.shape[0], img.shape[1]]

        targets = self.apply_pre_process(targets)

        if self.transforms is not None:
            targets['img'] = self.transforms(targets['img'])

        targets['text_polys'] = targets['text_polys'].tolist()

        if self.filter_key is not None and self.is_training:
            targets_dict = dict()
            for k, v in targets.items():
                if k not in self.filter_key:
                    targets_dict[k] = v
            return targets['img'], targets_dict
        else:
            return targets['img'], targets

    def get_annotation(self, gt_path):
        boxes = list()
        texts = list()
        ignores = list()
        with open(gt_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                # print(params)
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    # print(box)
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        texts.append(params[8])
                        ignores.append(params[8] in self.ignore_tags)
                except Exception as e:
                    print(e)
                    print('get annotation is failed {}'.format(gt_path))

        data = {'text_polys': np.array(boxes),
                'texts': texts,
                'ignore_tags': ignores}

        return data

    def init_pre_process(self):
        self.aug = list()
        if self.pre_processes is not None:
            for aug in self.pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def apply_pre_process(self, data):
        for aug in self.aug:
            data = aug(data)
        return data


class SynthTextData(object):
    def __init__(self, root, gt_name, pre_processes=None, transforms=None, filter_keys=None, ignore_tags=None,
                 is_training=True):
        self.is_training = is_training
        self.root = root
        self.gt_name = gt_name
        self.transforms = transforms
        self.pre_processes = pre_processes
        self.filter_key = filter_keys
        self.ignore_tags = ignore_tags

        # self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        # self.gts = list(sorted(os.listdir(os.path.join(root, "gts"))))
        self.synthtext = SynthTextDataGet(data_path=self.root, gt_name=self.gt_name)
        self.init_pre_process()

    def __len__(self):
        return self.synthtext.get_len()

    def __getitem__(self, item):
        img_path, rect_arr = self.synthtext.get_annotation_one(item)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        targets = self.get_annotation(rect_arr)
        targets['img'] = img
        targets['shape'] = [img.shape[0], img.shape[1]]

        targets = self.apply_pre_process(targets)

        if self.transforms is not None:
            targets['img'] = self.transforms(targets['img'])

        targets['text_polys'] = targets['text_polys'].tolist()

        if self.filter_key is not None and self.is_training:
            targets_dict = dict()
            for k, v in targets.items():
                if k not in self.filter_key:
                    targets_dict[k] = v
            return targets['img'], targets_dict
        else:
            return targets['img'], targets

    def get_annotation(self, rect_arr):
        boxes = list()
        texts = list()
        ignores = list()
        for idx in range(len(rect_arr)):
            params = rect_arr[idx].reshape(1, -1).tolist()[0]
            # print(params)
            try:
                box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                if cv2.contourArea(box) > 0:
                    boxes.append(box)
                    texts.append('111')
                    ignores.append('111' in self.ignore_tags)
            except Exception as e:
                print(e)
                print('get annotation is failed {}'.format(rect_arr))

        data = {'text_polys': np.array(boxes),
                'texts': texts,
                'ignore_tags': ignores}

        return data

    def init_pre_process(self):
        self.aug = list()
        if self.pre_processes is not None:
            for aug in self.pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def apply_pre_process(self, data):
        for aug in self.aug:
            data = aug(data)
        return data


if __name__ == '__main__':
    dataset = CurrentOcrData('/home/shizai/data2/ocr_data/rctw')
    dataset.get_annotation('/home/shizai/data2/ocr_data/rctw/gts/rctw_image_3629.txt')
