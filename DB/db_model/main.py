# -*- coding:utf-8 -*-
# @author :adolf
import sys
import torch
import detection_util.transforms as T

from detection_util.utils import collate_fn
from detection_util.engine import train_one_epoch
from db_model.data_define import *

import torch.utils.data as data

from db_model.model import DBModel
import os

from torchvision import transforms as T
from db_model.loss import DBLoss

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


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 2
    root_path = '/home/shizai/data2/ocr_data/rctw/'
    pre_processes = [{'type': 'IaaAugment', 'args': [{'type': 'Fliplr', 'args': {'p': 0.5}},
                                                     {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                                                     {'type': 'Resize', 'args': {'size': [0.5, 3]}}]},
                     {'type': 'EastRandomCropData', 'args': {'size': [640, 640], 'max_tries': 50, 'keep_ratio': True}},
                     {'type': 'MakeBorderMap', 'args': {'shrink_ratio': 0.4}},
                     {'type': 'MakeShrinkMap', 'args': {'shrink_ratio': 0.4, 'min_text_size': 8}}]
    transforms_config = [{'type': 'ToTensor', 'args': {}},
                         {'type': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}]
    filter_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags', 'shape', 'img']
    ignore_tags = ['*', '###']
    transforms = get_transforms(transforms_config)

    # print(transforms)

    dataset = CurrentOcrData(root=root_path,
                             pre_processes=pre_processes,
                             transforms=transforms,
                             filter_keys=filter_keys,
                             ignore_tags=ignore_tags)

    dataset_test = CurrentOcrData(root=root_path,
                                  pre_processes=pre_processes,
                                  transforms=transforms,
                                  filter_keys=filter_keys,
                                  ignore_tags=ignore_tags)

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    print('333333333', len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              shuffle=True, num_workers=4)  # ,
    # collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1, shuffle=False, num_workers=4)  # ,
    # collate_fn=collate_fn)

    model_config = {
        'backbone': 'deformable_resnet50',
        'pretrained': True,
        'segmentation_body': {'type': 'FPN', 'args': {'inner_channels': 256}},
        'segmentation_head': {'type': 'DBHead', 'args': {'out_channels': 2, 'k': 50}}
    }
    model = DBModel(model_config=model_config).to(device)
    # model.load_state_dict(torch.load('model_use/gen_20000.pth'))
    print(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=1e-7, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # def get_loss(config):
    #     return getattr(loss, config['type'])(**config['args'])

    criterion = DBLoss()

    num_epochs = 21

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, criterion, data_loader,
                        device, epoch, print_freq=10)
        lr_scheduler.step()
        if epoch % 5 == 0:
            # evaluate(ctpn_model, data_loader_test, device=device)
            torch.save(model.state_dict(),
                       'model_save/db_model_' + str(epoch) + '.pth')

    print("That's is all!")


if __name__ == '__main__':
    main()
