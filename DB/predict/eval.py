# -*- coding:utf-8 -*-
# @author :adolf
import os
import sys

import numpy as np
import argparse
import time
import torch
from tqdm.auto import tqdm
from detection_util.iou import DetectionIoUEvaluator

import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetric(object):
    def __init__(self):
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, is_output_polygon=False, box_thresh=0.5):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        gt_polyons_batch = batch['text_polys']
        # gt_polyons_batch = np.array([gt_polyons_batch])
        # print(gt_polyons_batch.shape)
        ignore_tags_batch = batch['ignore_tags']
        # ignore_tags_batch = np.array([ignore_tags_batch])
        # ignore_tags_batch = ignore_tags_batch.reshape(-1, 1)

        # print('pred_scores_batch', output[1])
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        # print('111', pred_polygons_batch.shape)
        # print('222', pred_scores_batch.shape)
        # print('333', gt_polyons_batch.shape)
        # print('444', ignore_tags_batch.shape)

        for polygons, pred_polygons, pred_scores, ignore_tags \
                in zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            # print('555', polygons.shape)
            # print('333', pred_polygons.shape)
            gt = [dict(points=polygons[i], ignore=ignore_tags[i]) for i in range(len(polygons))]
            if is_output_polygon:
                # print('123')
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                # print('456')
                pred = []
                # print('0000', pred_polygons.shape)
                for i in range(pred_polygons.shape[0]):
                    # print('pred_scores', pred_scores.shape)
                    # print(pred_scores[i])
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(dict(points=pred_polygons[i, :, :].tolist()))
            # print('=' * 50)
            # print('gt', gt)
            # print('pred', pred)
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        return self.measure(batch, output, is_output_polygon, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }


class EVAL(object):
    def __init__(self, model, model_path, validate_loader, post_process, metric_cls, gpu_id=0):
        self.device = torch.device("cuda:%s" % gpu_id)
        if gpu_id is not None:
            torch.backends.cudnn.benchmark = True
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # print(checkpoint)
        # config = checkpoint['config']
        # config['arch']['args']['pretrained'] = False

        self.validate_loader = validate_loader

        self.model = model
        # self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

        self.post_process = post_process
        self.metric_cls = metric_cls

    def eval(self):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, (img, batch) in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader),
                                    desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                h, w = batch['shape']
                batch['shape'] = [(h, w)]
                batch['image'] = img
                batch['text_polys'] = np.array([batch['text_polys']])
                batch['ignore_tags'] = np.array([batch['ignore_tags']])

                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                # print('image', batch['image'].shape)
                preds = self.model(batch['image'])
                # print(preds)
                boxes, scores = self.post_process.represent(batch, preds)
                # boxes, scores = boxes[0], scores[0]
                # print(scores)
                total_frame += batch['image'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


if __name__ == '__main__':
    # args = init_args()
    eval_config = {"model": None,
                   "model_path": None,
                   "validate_loader": None,
                   "post_process": None,
                   "metric_cls": QuadMetric}
    eval = EVAL(**eval_config)
    result = eval.eval()
    print(result)
