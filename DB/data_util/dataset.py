# -*- coding:utf-8 -*-
# @author :adolf
import os
import scipy.io as scio
import json
import numpy as np


class SynthTextDataGet(object):
    def __init__(self, data_path, gt_name):
        super(SynthTextDataGet, self).__init__()
        self.data_path = data_path
        self.gt_name = gt_name
        self.get_annotation()

    def get_annotation(self):
        gt_path = os.path.join(self.data_path, self.gt_name)
        data = scio.loadmat(gt_path)
        wordBB = data.get('wordBB')
        imnames = data.get('imnames')
        # print(imnames.shape)
        # print(wordBB.shape)
        MySynth = dict()
        MySynth['img'] = imnames
        MySynth['gt'] = wordBB

        # MySynth['img'] = MySynth['img'][:, :10]
        # MySynth['gt'] = MySynth['gt'][:, :10]

        self.MySynth = MySynth
        # print(MySynth)
        # scio.savemat('/home/shizai/data1/ocr_data/SynthText/gt_test.mat', MySynth)
        return self.MySynth

    def get_annotation_one(self, index):
        # self.get_annotation()
        img_name = self.MySynth['img'][:, index][0][0]
        img_path = os.path.join(self.data_path, img_name)
        # print(img_path)
        # gt_arr = self.MySynth['gt'][:, index]

        rec = np.array(self.MySynth['gt'][0, index], dtype=np.int32)
        if len(rec.shape) == 3:
            rec = rec.transpose(2, 1, 0)
            # print('111', rec)
        else:
            rec = rec.transpose(1, 0)[np.newaxis, :]
            # print('222', rec)
        return img_path, rec

    def get_len(self):
        # print(self.MySynth['img'].shape[1])
        return self.MySynth['img'].shape[1]


if __name__ == '__main__':
    synthtext = SynthTextDataGet(data_path='/home/shizai/data1/ocr_data/SynthText', gt_name='gt.mat')
    synthtext.get_len()
    # with open('/home/shizai/data1/ocr_data/SynthText/gt.json', 'a') as outfile:
    #     json.dump(gt_dict, outfile, ensure_ascii=False)
    #     outfile.write('\n')
