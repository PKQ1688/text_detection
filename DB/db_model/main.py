# -*- coding:utf-8 -*-
# @author :adolf
import sys
import torch
import detection_util.transforms as T

from detection_util.utils import icdar_collate_fn, collate_fn
from detection_util.engine import train_one_epoch
from db_model.data_define import *

import torch.utils.data as data

from db_model.model import DBModel
import os

from torchvision import transforms as T
from db_model.loss import DBLoss

import torch.nn as nn
from predict.eval import *

from post_processing.seg_detector_representer import SegDetectorRepresenter

from apex import amp

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,4,6"
use_amp = False
BATCH_SIZE = 16


# import apex


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
    root_path = '/home/shizai/data2/ocr_data/icdar2015/train/'
    # root_path = '/home/shizai/data1/ocr_data/SynthText/'

    # root_path = "/home/shizai/data2/ocr_data/third_data/"

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
                             ignore_tags=ignore_tags,
                             is_training=True)
    #
    # dataset_test = CurrentOcrData(root=root_path,
    #                               pre_processes=pre_processes,
    #                               transforms=transforms,
    #                               filter_keys=filter_keys,
    #                               ignore_tags=ignore_tags,
    #                               is_training=False)

    # dataset = SynthTextData(root=root_path,
    #                         gt_name='gt.mat',
    #                         pre_processes=pre_processes,
    #                         transforms=transforms,
    #                         filter_keys=filter_keys,
    #                         ignore_tags=ignore_tags,
    #                         is_training=True)

    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    print('333333333', len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=0)  # , collate_fn=icdar_collate_fn)

    # data_loader_test = torch.utils.data.DataLoader(dataset_test,
    #                                                batch_size=1, shuffle=False, num_workers=4)  # ,
    # collate_fn=collate_fn)

    model_config = {
        'backbone': 'resnet50',
        'pretrained': True,
        'segmentation_body': {'type': 'FPN', 'args': {'inner_channels': 256}},
        'segmentation_head': {'type': 'DBHead', 'args': {'out_channels': 2, 'k': 50}}
    }
    model = DBModel(model_config=model_config)

    model = model.float()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=0.0005)

    # optimizer = torch.optim.Adam(params, lr=1e-7, weight_decay=0.0005)
    if use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.DataParallel(model)

    # model.load_state_dict(torch.load('model_use/gen_20000.pth'))
    # print(model)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)

    # def get_loss(config):
    #     return getattr(loss, config['type'])(**config['args'])

    criterion = DBLoss()

    params_config = dict()
    params_config['thresh'] = 0.5
    params_config['box_thresh'] = 0.5
    params_config['max_candidates'] = 1000
    params_config['unclip_ratio'] = 1.5
    num_epochs = 1200
    # post_process = SegDetectorRepresenter(thresh=params_config['thresh'],
    #                                       box_thresh=params_config['box_thresh'],
    #                                       max_candidates=params_config['max_candidates'],
    #                                       unclip_ratio=params_config['unclip_ratio'])
    min_loss = 9999
    for epoch in range(num_epochs):
        try:
            # print(epoch)
            losses = train_one_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq=50,
                                     use_amp=use_amp)
            lr_scheduler.step()
            if epoch > num_epochs / 2 and losses < min_loss:
                torch.save(model.state_dict(),
                           'model_save/db_loss_model_' + str(epoch) + '.pth')
                min_loss = losses

        except Exception as e:
            print('111')
            print(e)

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       'model_save/db_icdar_model_' + str(epoch) + '.pth')

            # eval_config = {"model": model,
            #                "model_path": 'model_save/final.pth',
            #                "validate_loader": data_loader_test,
            #                "post_process": post_process,
            #                "metric_cls": QuadMetric()}
            #
            # eval_ = EVAL(**eval_config)
            # recall, precision, fmeasure = eval_.eval()
            # print("recall:", recall)
            # print("precision:", precision)
            # print("fmeasure:", fmeasure)

    torch.save(model.state_dict(), 'model_save/db_icdar_model_final.pth')

    print("That's is all!")


if __name__ == '__main__':
    main()
