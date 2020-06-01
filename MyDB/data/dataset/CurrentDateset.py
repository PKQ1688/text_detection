# -*- coding:utf-8 -*-
# @author :adolf
import os

from data.data_utils import order_points_clockwise
from data.data_aug import *
from data.make_labels import *

from tqdm import tqdm
from torchvision import transforms as T
import time

cv2.setNumThreads(0)


# cv2.ocl.setUseOpenCL(False)


class CurrentOcrData(object):
    def __init__(self, root, pre_processes=None, transforms=None, filter_keys=None, ignore_tags=None, is_training=True):
        self.is_training = is_training
        self.root = root
        self.transforms = transforms
        self.pre_processes = pre_processes
        self.filter_key = filter_keys
        self.ignore_tags = ignore_tags

        self.aug = list()

        self.patients = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        # self.imgs = self.imgs[:30]
        # self.gts = list(sorted(os.listdir(os.path.join(root, "gts"))))
        # self.gts = self.gts[:30]
        validation_cases = int(0.1 * len(self.patients))
        #
        validation_patients = random.sample(self.patients, k=validation_cases)
        # validation_patients = ['img_363.jpg']

        if not is_training:
            self.patients = validation_patients
        else:
            self.patients = sorted(
                list(set(self.patients).difference(validation_patients))
            )
        # print(self.imgs)
        self.init_pre_process()
        # print(self.aug)
        # self.targets = list()
        # for img_name in tqdm(self.patients):
        #     img_path = os.path.join(self.root, 'imgs', img_name)
        #     gt_name = 'gt_' + img_name.replace('png', 'txt').replace('jpg', 'txt').replace('jpeg', 'txt')
        #     gt_path = os.path.join(self.root, 'gts', gt_name)
        #
        #     img = cv2.imread(img_path)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        #     one_targets = self.get_annotation(gt_path)
        #     one_targets['img'] = img
        #     one_targets['shape'] = [img.shape[0], img.shape[1]]
        #
        #     self.targets.append(one_targets)

    def __len__(self):
        return len(self.patients)  # - 1

    def __getitem__(self, item):
        # s1 = time.time()
        img_path = os.path.join(self.root, 'imgs', self.patients[item])
        gt_name = self.patients[item].replace('png', 'txt').replace('jpg', 'txt').replace('jpeg', 'txt')
        gt_path = os.path.join(self.root, 'gts', gt_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # e_5 = time.time()
        # print('read img time:', e_5 - s1)

        one_targets = self.get_annotation(gt_path)
        one_targets['img'] = img
        one_targets['shape'] = [img.shape[0], img.shape[1]]
        #
        targets = one_targets
        # targets = self.targets[item]
        # s2 = time.time()
        if self.is_training:
            targets = self.apply_pre_process(targets)
            # e2 = time.time()
            # print('aug use time:', e2 - s2)
        else:
            val_aug = list()
            for aug in self.pre_processes:
                if aug['type'] not in ['MakeBorderMap', 'MakeShrinkMap']:
                    continue
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                val_aug.append(cls)
            for aug in val_aug:
                targets = aug(targets)

        # s3 = time.time()
        if self.transforms is not None:
            targets['img'] = self.transforms(targets['img'])

        targets['text_polys'] = targets['text_polys'].tolist()
        # e3 = time.time()
        # print('transforms time:', e3 - s3)
        if self.filter_key is not None and self.is_training:
            targets_dict = dict()
            for k, v in targets.items():
                if k not in self.filter_key:
                    targets_dict[k] = v
            # end_time = time.time()
            # print('train one image use time:', end_time - s1)
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
    import yaml
    from torch.utils.data import DataLoader
    import torch

    with open('config/db_resnet50.yaml', 'r') as fp:
        configs = yaml.load(fp.read(), Loader=yaml.FullLoader)


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


    image_path = '/home/shizai/data2/ocr_data/rctw'

    pre_processes = configs['data']['process']
    transforms_config = configs['data']['transforms']
    transforms = get_transforms(transforms_config)
    filter_keys = configs['data']['filter_keys']
    ignore_tags = configs['data']['ignore_tags']

    train_datasets = CurrentOcrData(root=image_path,
                                    pre_processes=pre_processes,
                                    transforms=transforms,
                                    filter_keys=filter_keys,
                                    ignore_tags=ignore_tags,
                                    is_training=True)
    loader_train = DataLoader(
        train_datasets,
        batch_size=16,
        shuffle=True,
        drop_last=False,
        num_workers=32,
        pin_memory=False
    )
    # for i in range(1000):
    #     print(i)
    # _ = train_datasets.__getitem__(i)

    s = time.time()
    flag_time = s
    for i, data in enumerate(loader_train):
        one_time = time.time()
        print('use one time:', one_time - flag_time)
        flag_time = one_time
    #
    e = time.time()
    print('cost time', e - s)
