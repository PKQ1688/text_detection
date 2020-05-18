# -*- coding:utf-8 -*-
# @author :adolf
import os
import torch
from data.dataset import CurrentOcrData
from torchvision import transforms as T

from torch.utils.data import DataLoader
import torch.optim as optim

from model.dbmodel_v2 import dbnet_resnet50_fpn
# from model.dbmodel import DBModel
from model.loss import DBLoss
import torch.nn as nn

from utils import lr_scheduler

from tqdm import tqdm

from tools.eval import EvalChinaLife

from prefetch_generator import BackgroundGenerator
import numpy as np
import shutil

import warnings

import time

import cv2

warnings.filterwarnings('ignore')

# torch.multiprocessing.set_start_method('spawn')

# torch.set_num_threads(8)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.multiprocessing.set_sharing_strategy('file_system')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            # self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            # self.next_target = {key: self.next_target[key].cuda(non_blocking=True) for key in self.next_target}
            for key, value in self.next_target.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        self.next_target[key] = value.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class Train(object):
    def __init__(self, configs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.image_path = configs['root_path']['image_path']
        self.eval_path = configs['root_path']['eval_path']

        self.pre_processes = configs['data']['process']
        transforms_config = configs['data']['transforms']
        self.transforms = self.get_transforms(transforms_config)
        self.filter_keys = configs['data']['filter_keys']
        self.ignore_tags = configs['data']['ignore_tags']

        self.batch_size = configs['train_params']['batch_size']
        self.epochs = configs['train_params']['epochs']
        self.lr = configs['train_params']['lr']
        self.workers = configs['train_params']['workers']
        is_multi_gpu = configs['train_params']['DateParallel']

        self.weights = configs['root_path']['weight']
        if os.path.exists(self.weights):
            shutil.rmtree(self.weights)
            os.mkdir(self.weights)
        else:
            os.mkdir(self.weights)

        pre_train = configs['pretrained']['pretrained_model']
        pre_train_backbone = configs['pretrained']['pretrained_backbone']

        model_path = configs['root_path']['pretrained_model']

        self.model = dbnet_resnet50_fpn(pre_train_backbone)
        # self.model = DBModel(pre_train_backbone)
        if pre_train:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

        if is_multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        # self.best_hmean = 0.0
        self.best_val_loss = 999

        # self.loader_train, self.loader_valid, self.train_prefetcher, self.valid_prefetcher = self.data_loaders()
        self.loader_train, self.loader_valid = self.data_loaders()

        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=0.0005)
        # self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.0005)

        self.scheduler = lr_scheduler.LR_Scheduler_Head(mode='poly',
                                                        base_lr=self.lr,
                                                        num_epochs=self.epochs,
                                                        iters_per_epoch=len(self.loader_train),
                                                        warmup_epochs=1)

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.patience = 0

        self.criterion = DBLoss()

        self.step = 0

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
        train_datasets = CurrentOcrData(root=self.image_path,
                                        pre_processes=self.pre_processes,
                                        transforms=self.transforms,
                                        filter_keys=self.filter_keys,
                                        ignore_tags=self.ignore_tags,
                                        is_training=True)
        # valid_datasets = train_datasets
        #
        valid_datasets = CurrentOcrData(root=self.image_path,
                                        pre_processes=self.pre_processes,
                                        transforms=self.transforms,
                                        filter_keys=self.filter_keys,
                                        ignore_tags=self.ignore_tags,
                                        is_training=False)
        return train_datasets, valid_datasets

    def data_loaders(self):
        dataset_train, dataset_valid = self.datasets()
        loader_train = DataLoaderX(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.workers,
            pin_memory=False
        )
        # train_prefetcher = data_prefetcher(loader_train)

        loader_valid = DataLoaderX(
            dataset_valid,
            batch_size=1,
            drop_last=False,
            num_workers=self.workers,
        )
        # valid_prefetcher = data_prefetcher(loader_valid)

        return loader_train, loader_valid  # , train_prefetcher, valid_prefetcher

    def train_one_epoch(self, epoch):

        self.model.train()
        loss_train = []
        one_epoch_use_time = 0
        one_epoch_data_process_use_time = 0
        for i, data in enumerate(self.loader_train):
            # img, targets = self.train_prefetcher.next()
            # i = 0
            # while img is not None:
            #     i += 1

            self.scheduler(self.optimizer, i, epoch, self.best_val_loss)
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as profile:
            data_process_start_time = time.time()

            img, targets = data
            img = img.to(self.device)
            # print(img.shape)
            # print(img)
            # print(targets.keys())
            for key, value in targets.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        targets[key] = value.to(self.device)

            data_process_end_time = time.time()
            one_epoch_data_process_use_time += data_process_end_time - data_process_start_time
            # print('data_process time:', profile)

            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as profile:

            torch.cuda.synchronize()
            train_start_time = time.time()

            y_pred = self.model(img)
            # print('1111', y_pred.size())
            # print('2222', y_true.size())
            loss_dict = self.criterion(y_pred, targets)
            loss = loss_dict['loss']
            loss_train.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.synchronize()
            train_end_time = time.time()
            one_epoch_use_time += train_end_time - train_start_time
            # print('train_use time', profile)

            # lr_scheduler.step()
            if self.step % 50 == 0:
                print('Epoch:[{}/{}]\t iter:[{}]\t loss={:.5f}\t lr={}'
                      .format(epoch, self.epochs, i, loss, self.optimizer.param_groups[0]['lr']))

            self.step += 1
            # img, targets = self.train_prefetcher.next()
        print("Epoch [{}] train use time:[{:.2f}]".format(epoch, one_epoch_use_time))
        print("Epoch [{}] data_process use time:[{:.2f}]".format(epoch, one_epoch_data_process_use_time))

    def val_model(self, epoch=0):
        self.model.eval()

        loss_val = []

        # img, targets = self.valid_prefetcher.next()
        # i = 0
        # while img is not None:
        #     i += 1
        for i, data in enumerate(self.loader_valid):
            img, targets = data

            img = img.to(self.device)

            for key, value in targets.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        targets[key] = value.to(self.device)

            # print(x.size())
            # print(333,x[0][2]
            # print('test input tensor size', x.size())
            # print('test input tensor', x)
            with torch.no_grad():
                y_pred = self.model(img)
                # print('y_pred_shape', y_pred.size())
                # print('y_pred', y_pred)
                # print('targets', targets.keys())
                loss_dict = self.criterion(y_pred, targets)
                loss = loss_dict['loss']

                loss_val.append(loss.item())
            # img, targets = self.valid_prefetcher.next()

        loss_mean = np.mean(np.array(loss_val))
        print('mean val loss:{:.5f}'.format(loss_mean))
        if loss_mean < self.best_val_loss:
            self.best_val_loss = loss_mean
            self.patience = 0
            use_model = os.path.join(self.weights, "DB_{}_{:.2f}.pth".format(epoch, loss_mean))
            torch.save(self.model.state_dict(), use_model)
        else:
            self.patience += 1

    def main(self):
        # eval_path = '/home/shizai/data2/ocr_data/china_life_test_data'
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            epoch_start_time = time.time()
            self.train_one_epoch(epoch)
            epoch_end_time = time.time()
            print("Epoch [{}] all use time:[{:.2f}]".format(epoch, epoch_end_time - epoch_start_time))
            # self.lr_scheduler.step()
            self.val_model(epoch)
            # if self.patience > 50:
            #     break

            # if epoch % 2 == 1:
            #     use_model = os.path.join(self.weights, "DB_{}.pth".format(epoch))
            #     print(use_model)
            #     torch.save(self.model.state_dict(), use_model)
            #     _, _, hmean = EvalChinaLife(self.eval_path, use_model).main()
            #     if hmean > self.best_hmean:
            #         self.best_hmean = hmean
            #     else:
            #         os.remove(use_model)

        torch.save(self.model.state_dict(), os.path.join(self.weights, "DB_final.pth"))


if __name__ == '__main__':
    import yaml

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"

    with open('config/db_resnet50.yaml', 'r') as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)

    trainer = Train(configs=config)
    trainer.main()
