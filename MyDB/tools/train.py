# -*- coding:utf-8 -*-
# @author :adolf
import torch
from data.dataset import CurrentDateset
from torchvision import transforms as T

import torch.utils.data as data


class Train(object):
    def __init__(self, configs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.image_path = configs['root_path']['image_path']

        self.pre_processes = configs['data']['process']
        transforms_config = configs['data']['transforms']
        self.transforms = self.get_transforms(transforms_config)
        self.filter_keys = configs['data']['filter_keys']
        self.ignore_tags = configs['data']['ignore_tags']

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

    def datasets(self):
        train_datasets = CurrentDateset(root=self.image_path,
                                        pre_processes=self.pre_processes,
                                        transforms=self.transforms,
                                        filter_keys=self.filter_keys,
                                        ignore_tags=self.ignore_tags,
                                        is_training=True)
        # valid_datasets = train_datasets

        valid_datasets = CurrentDateset(root=self.root_path,
                                        pre_processes=self.pre_processes,
                                        transforms=self.transforms,
                                        filter_keys=self.filter_keys,
                                        ignore_tags=self.ignore_tags,
                                        is_training=False)
        return train_datasets, valid_datasets
