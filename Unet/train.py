# -*- coding:utf-8 -*-
# @author :adolf
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UNetSegmentationDataset as Dataset
from loss import DiceLoss
from data_aug.transforms import get_transforms
from unet_model import UNet
from utils import dsc
import logging

# from pytorchtools import EarlyStopping

import cv2
import lr_scheduler

import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class Train(object):
    def __init__(self, configs):
        self.batch_size = configs.get("batch_size", "16")
        self.epochs = configs.get("epochs", "100")
        self.lr = configs.get("lr", "0.0001")

        device_args = configs.get("device", "cuda")
        self.device = torch.device("cpu" if not torch.cuda.is_available() else device_args)

        self.workers = configs.get("workers", "4")

        self.vis_images = configs.get("vis_images", "200")
        self.vis_freq = configs.get("vis_freq", "10")

        self.weights = configs.get("weights", "./weights")
        if not os.path.exists(self.weights):
            os.mkdir(self.weights)

        self.logs = configs.get("logs", "./logs")
        if not os.path.exists(self.weights):
            os.mkdir(self.weights)

        self.images_path = configs.get("images_path", "./data")

        self.is_resize = config.get("is_resize", False)
        self.image_short_side = config.get("image_short_side", 256)

        self.is_padding = config.get("is_padding", False)

        is_multi_gpu = config.get("DateParallel", False)

        pre_train = config.get("pre_train", False)
        model_path = config.get("model_path", './weights/unet_idcard_adam.pth')

        # self.image_size = configs.get("image_size", "256")
        # self.aug_scale = configs.get("aug_scale", "0.05")
        # self.aug_angle = configs.get("aug_angle", "15")

        self.step = 0

        self.dsc_loss = DiceLoss()
        self.model = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        if pre_train:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

        if is_multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        self.best_validation_dsc = 0.0

        self.loader_train, self.loader_valid = self.data_loaders()

        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=0.0005)
        # self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.scheduler = lr_scheduler.LR_Scheduler_Head('poly', self.lr,
                                                        self.epochs, len(self.loader_train))

    def datasets(self):
        train_datasets = Dataset(images_dir=self.images_path,
                                 # image_size=self.image_size,
                                 subset="train",  # train
                                 transform=get_transforms(train=True),
                                 is_resize=self.is_resize,
                                 image_short_side=self.image_short_side,
                                 is_padding=self.is_padding
                                 )
        # valid_datasets = train_datasets

        valid_datasets = Dataset(images_dir=self.images_path,
                                 # image_size=self.image_size,
                                 subset="validation",  # validation
                                 transform=get_transforms(train=False),
                                 is_resize=self.is_resize,
                                 image_short_side=self.image_short_side,
                                 is_padding=False
                                 )
        return train_datasets, valid_datasets

    def data_loaders(self):
        dataset_train, dataset_valid = self.datasets()

        loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.workers,
        )
        loader_valid = DataLoader(
            dataset_valid,
            batch_size=1,
            drop_last=False,
            num_workers=self.workers,
        )

        return loader_train, loader_valid

    @staticmethod
    def dsc_per_volume(validation_pred, validation_true):
        assert len(validation_pred) == len(validation_true)
        dsc_list = []
        for p in range(len(validation_pred)):
            y_pred = np.array([validation_pred[p]])
            y_true = np.array([validation_true[p]])
            dsc_list.append(dsc(y_pred, y_true))
        return dsc_list

    @staticmethod
    def get_logger(filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def train_one_epoch(self, epoch):

        self.model.train()
        loss_train = []
        for i, data in enumerate(self.loader_train):
            self.scheduler(self.optimizer, i, epoch, self.best_validation_dsc)
            x, y_true = data
            x, y_true = x.to(self.device), y_true.to(self.device)

            y_pred = self.model(x)
            # print('1111', y_pred.size())
            # print('2222', y_true.size())
            loss = self.dsc_loss(y_pred, y_true)

            loss_train.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # lr_scheduler.step()
            if self.step % 200 == 0:
                print('Epoch:[{}/{}]\t iter:[{}]\t loss={:.5f}\t '.format(epoch, self.epochs, i, loss))

            self.step += 1

    def eval_model(self, patience):
        self.model.eval()
        loss_valid = []

        validation_pred = []
        validation_true = []
        # early_stopping = EarlyStopping(patience=patience, verbose=True)

        for i, data in enumerate(self.loader_valid):
            x, y_true = data
            x, y_true = x.to(self.device), y_true.to(self.device)

            # print(x.size())
            # print(333,x[0][2])
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.dsc_loss(y_pred, y_true)

            # print(y_pred.shape)
            mask = y_pred > 0.5
            mask = mask * 255
            mask = mask.cpu().numpy()[0][0]
            # print(mask)
            # print(mask.shape())
            cv2.imwrite('result.png', mask)

            loss_valid.append(loss.item())

            y_pred_np = y_pred.detach().cpu().numpy()

            validation_pred.extend(
                [y_pred_np[s] for s in range(y_pred_np.shape[0])]
            )
            y_true_np = y_true.detach().cpu().numpy()
            validation_true.extend(
                [y_true_np[s] for s in range(y_true_np.shape[0])]
            )

        # early_stopping(loss_valid, self.model)
        # if early_stopping.early_stop:
        #     print('Early stopping')
        #     import sys
        #     sys.exit(1)
        mean_dsc = np.mean(
            self.dsc_per_volume(
                validation_pred,
                validation_true,
            )
        )
        # print('mean_dsc:', mean_dsc)
        if mean_dsc > self.best_validation_dsc:
            self.best_validation_dsc = mean_dsc
            torch.save(self.model.state_dict(), os.path.join(self.weights, "unet_xia_adam.pth"))
            print("Best validation mean DSC: {:4f}".format(self.best_validation_dsc))

    def main(self):
        # print('train is begin.....')
        # print('load data end.....')

        # loaders = {"train": loader_train, "valid": loader_valid}

        for epoch in tqdm(range(self.epochs), total=self.epochs):
            self.train_one_epoch(epoch)
            self.eval_model(patience=10)

        torch.save(self.model.state_dict(), os.path.join(self.weights, "unet_final.pth"))


if __name__ == '__main__':
    import yaml

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5,6"

    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)

    trainer = Train(configs=config)
    trainer.main()
