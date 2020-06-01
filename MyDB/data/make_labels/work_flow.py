# -*- coding:utf-8 -*-
# @author :adolf
from data.make_labels.make_border_map import MakeBorderMap
from data.make_labels.make_shrink_map import MakeShrinkMap
import numpy as np
from data.data_utils.clockwise_points import order_points_clockwise

import cv2
import os


# img_path = '/home/shizai/data2/ocr_data/rctw/imgs/rctw_image_3890.jpg'
# gt_path = '/home/shizai/data2/ocr_data/rctw/gts/rctw_image_3890.txt'


def get_annotation(gt_path, ignore_tags=['*', '###']):
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
                    ignores.append(params[8] in ignore_tags)
            except Exception as e:
                print(e)
                print('get annotation is failed {}'.format(gt_path))

    data = {'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores}

    return data


# data = get_annotation(gt_path)

# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# data['img'] = img
# print(data['ignore_tags'])
# data = MakeShrinkMap()(data)

# cv2.imwrite('images_result/ori_img.png', img)
# print(data['shrink_map'])

# data = MakeBorderMap()(data)
# print(data.keys())
# cv2.imwrite('images_result/shrink_map.png', (data['shrink_map'] * 255).astype(np.uint8))
# cv2.imwrite('images_result/shrink_mask.png', (data['shrink_mask'] * 255).astype(np.uint8))

# cv2.imwrite('images_result/threshold_map.png', (data['threshold_map'] * 255).astype(np.uint8))
# cv2.imwrite('images_result/threshold_mask.png', (data['threshold_mask'] * 255).astype(np.uint8))

def make_use_label(file_path, img_name):
    img_path = os.path.join(file_path, 'imgs', img_name)
    gt_name = 'gt_' + img_name.replace('png', 'txt').replace('jpg', 'txt').replace('jpeg', 'txt')
    gt_path = os.path.join(file_path, 'gts', gt_name)

    data = get_annotation(gt_path)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data['img'] = img

    data = MakeShrinkMap()(data)
    data = MakeBorderMap()(data)

    cv2.imwrite(os.path.join(file_path, 'shrink_map', img_name), data['shrink_map'])
    cv2.imwrite(os.path.join(file_path, 'shrink_mask', img_name), data['shrink_mask'])
    #
    cv2.imwrite(os.path.join(file_path, 'threshold_map', img_name), data['threshold_map'])
    cv2.imwrite(os.path.join(file_path, 'threshold_mask', img_name), data['threshold_mask'])


rctw_path = "/home/shizai/data2/ocr_data/rctw"
rctw_list = os.listdir(os.path.join(rctw_path, 'imgs'))

# print('begin...')
# for rctw_img in rctw_list:
#     make_use_label(rctw_path, rctw_img)
#
# third_path = "/home/shizai/data2/ocr_data/third_data"
# third_list = os.listdir(os.path.join(third_path, 'imgs'))
#
# for third_img in third_list:
#     make_use_label(third_path, third_img)
# print('end...')

icdar_path = "/home/shizai/data2/ocr_data/icdar2015/train/"
icdar_list = os.listdir(os.path.join(icdar_path, 'imgs'))

for icdar_img in icdar_list:
    make_use_label(icdar_path, icdar_img)
