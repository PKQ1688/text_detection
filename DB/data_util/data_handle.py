# -*- coding:utf-8 -*-
# @author :adolf
import os
import shutil
import json

Root = '/home/shizai/data2/ocr_data/实在智能科技自然场景数据交付/'
Dest = '/home/shizai/data2/ocr_data/third_data/'


def get_file(Root):
    for (root, dirs, files) in os.walk(Root):
        for f in files:
            shotname, extension = os.path.splitext(f)
            # print(shotname)
            # print(extension)
            if extension in ['.jpg', '.png', '.json']:
                old_path = os.path.join(root, f)
                new_path = os.path.join(Dest, f)
                # print('=====', old_path)
                # print('+++++', new_path)
                shutil.copyfile(old_path, new_path)


def json_txt(json_path):
    with open(json_path, 'r') as fp:
        json_re = json.load(fp)
    # print(json_re.keys())
    masks = json_re['marks']
    gt_path = json_path.replace('/json/', '/gt/').replace('.json', '.txt')
    print(gt_path)
    with open(gt_path, 'w') as ff:
        for one_polygon in masks:
            points = one_polygon['point']
            for point in points:
                # print(point)
                ff.write(str(point['x']))
                ff.write(',')
                ff.write(str(point['y']))
                ff.write(',')

            ff.write('text')
            ff.write('\n')


# json_txt('/home/shizai/data2/ocr_data/third_data/json/IMG_20200115_182649.json')

json_list = os.listdir('/home/shizai/data2/ocr_data/third_data/json/')
for json_name in json_list:
    json_path = os.path.join('/home/shizai/data2/ocr_data/third_data/json/', json_name)
    json_txt(json_path)
