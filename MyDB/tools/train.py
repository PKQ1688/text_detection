# -*- coding:utf-8 -*-
# @author :adolf
import os
import torch
from data.dataset import CurrentOcrData
from torchvision import transforms as T

from torch.utils.data import DataLoader
import torch.optim as optim

# from model.dbmodel import dbnet_resnet50_fpn
from model.dbmodel import DBModel
from model.loss import DBLoss
import torch.nn as nn

from utils import lr_scheduler

from tqdm import tqdm

from tools.eval import EvalChinaLife

from prefetch_generator import BackgroundGenerator

torch.multiprocessing.set_start_method('spawn')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Train(object):
    def __init__(self, configs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.image_path = configs['root_path']['image_path']
        self.eval_path = config['root_path']['eval_path']

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
        if not os.path.exists(self.weights):
            os.mkdir(self.weights)

        pre_train = configs['pretrained']['pretrained_model']
        pre_train_backbone = configs['pretrained']['pretrained_backbone']

        model_path = configs['root_path']['pretrained_model']

        # self.model = dbnet_resnet50_fpn(pre_train_backbone)
        self.model = DBModel(pre_train_backbone)
        if pre_train:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

        if is_multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        self.best_hmean = 0.0

        self.loader_train, self.loader_valid = self.data_loaders()

        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=0.0005)
        # self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.0005)

        self.scheduler = lr_scheduler.LR_Scheduler_Head('cos', self.lr,
                                                        self.epochs, len(self.loader_train))

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

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
        valid_datasets = train_datasets
        #
        # valid_datasets = CurrentOcrData(root=self.image_path,
        #                                 pre_processes=self.pre_processes,
        #                                 transforms=self.transforms,
        #                                 filter_keys=self.filter_keys,
        #                                 ignore_tags=self.ignore_tags,
        #                                 is_training=False)
        return train_datasets, valid_datasets

    def data_loaders(self):
        dataset_train, dataset_valid = self.datasets()
        loader_train = DataLoaderX(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.workers,
            # pin_memory=True
        )
        loader_valid = DataLoaderX(
            dataset_valid,
            batch_size=1,
            drop_last=False,
            num_workers=self.workers,
        )

        return loader_train, loader_valid

    def train_one_epoch(self, epoch):

        self.model.train()
        loss_train = []
        for i, data in enumerate(self.loader_train):
            self.scheduler(self.optimizer, i, epoch, self.best_hmean)
            x, targets = data
            x = x.to(self.device)
            # print(x.shape)
            # print(x)
            # print(targets.keys())
            for key, value in targets.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        targets[key] = value.to(self.device)

            y_pred = self.model(x)
            # print('1111', y_pred.size())
            # print('2222', y_true.size())
            loss_dict = self.criterion(y_pred, targets)
            loss = loss_dict['loss']
            loss_train.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # lr_scheduler.step()
            if self.step % 100 == 0:
                print('Epoch:[{}/{}]\t iter:[{}]\t loss={:.5f}\t '.format(epoch, self.epochs, i, loss))
                print(self.optimizer.param_groups[0]['lr'])

            self.step += 1

    # def eval_model(self, patience):
    # self.model.eval()
    #     loss_valid = []
    #
    #     validation_pred = []
    #     validation_true = []
    #     # early_stopping = EarlyStopping(patience=patience, verbose=True)
    #
    #     for i, data in enumerate(self.loader_valid):
    #         x, y_true = data
    #         x, y_true = x.to(self.device), y_true.to(self.device)
    #
    #         # print(x.size())
    #         # print(333,x[0][2])
    #         with torch.no_grad():
    #             y_pred = self.model(x)
    #             loss = self.dsc_loss(y_pred, y_true)
    #
    #         # print(y_pred.shape)
    #         pass

    def main(self):
        # eval_path = '/home/shizai/data2/ocr_data/china_life_test_data'
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            self.train_one_epoch(epoch)
            # self.lr_scheduler.step()
            # self.eval_model(patience=10)

            if epoch % 2 == 1:
                use_model = os.path.join(self.weights, "DB_{}.pth".format(epoch))
                print(use_model)
                torch.save(self.model.state_dict(), use_model)
                _, _, hmean = EvalChinaLife(self.eval_path, use_model).main()
                if hmean > self.best_hmean:
                    self.best_hmean = hmean
                else:
                    os.remove(use_model)

        torch.save(self.model.state_dict(), os.path.join(self.weights, "DB_final.pth"))


if __name__ == '__main__':
    import yaml

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    with open('config/db_resnet50.yaml', 'r') as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)

    trainer = Train(configs=config)
    trainer.main()
