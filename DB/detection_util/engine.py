# -*- coding:utf-8 -*-
# @author :adolf
import math
import sys
import time
import torch

import detection_util.utils as utils


def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch:[{}]'.format(epoch)

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    train_loss = 0

    # for batch_idx, (images, targets) in enumerate(data_loader):
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # images = list(torch.Tensor(image).to(device) for image in images)
        # targets = [{k: torch.Tensor(v).to(device) for k, v in t.items()} for t in targets]

        # print(images)
        # print(targets)
        # print(len(targets))
        # sys.exit(1)
        images = images.to(device)
        for key, value in targets.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(device)

        preds = model(images)
        loss_dict = criterion(preds, targets)

        # print('111', loss_dict)
        # sys.exit(1)

        losses = loss_dict['loss']
        # print('222', losses)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # print('333', loss_dict_reduced)
        # #
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print('444', losses_reduced)

        # loss_value = losses_reduced.item()
        #
        # if not math.isfinite(loss_value):
        #     targets_ = targets[0]
        #     img_id = targets_['image_id'].item()
        #     print('11111', img_id)
        # img_data_dict = data_loader.dataset.__dict__
        # img_data_dataset = img_data_dict['dataset']
        # img_data_dataset_dict = img_data_dataset.__dict__
        #
        # img_list = img_data_dataset_dict['imgs']
        # print('222222', img_list[img_id])
        #
        # print("Loss is {}, stopping training".format(loss_value))
        # sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        break

        # loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
        # for idx, (key, value) in enumerate(loss_dict.items()):
        #     loss_dict[key] = value.item()
        #     if key == 'loss':
        #         continue
        #     loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
        #     if idx < len(loss_dict) - 1:
        #         loss_str += ', '
        #
        # train_loss += loss_dict['loss']

        # print(epoch, train_loss)
