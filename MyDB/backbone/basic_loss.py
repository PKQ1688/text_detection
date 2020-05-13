# -*- coding:utf-8 -*-
# @author :adolf
import torch
import torch.nn as nn

import torch.nn.functional as F


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class GaussianMapLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(GaussianMapLoss, self).__init__()
        self.eps = eps

    def forward(self, gaussian_map, gaussian_gt, training_mask, border_map=None, text_thres=0.2, center_thres=0.7):
        """
        Weighted MSE-loss
        Args:
            gaussian_map: gaussian_map from network outputs
            gaussian_gt: gt for gaussian_map
            training_mask:
        """
        gaussian_map = torch.sigmoid(gaussian_map)
        text_map = torch.where(gaussian_gt > text_thres, torch.ones_like(gaussian_gt), torch.zeros_like(gaussian_gt))
        center_map = torch.where(gaussian_gt > center_thres, torch.ones_like(gaussian_gt),
                                 torch.zeros_like(gaussian_gt))
        center_gt = torch.where(gaussian_gt > center_thres, gaussian_gt, torch.zeros_like(gaussian_gt))
        text_gt = torch.where(gaussian_gt > text_thres, gaussian_gt, torch.zeros_like(gaussian_gt))
        bg_map = 1. - text_map

        pos_num = torch.sum(text_map)
        neg_num = torch.sum(bg_map)

        pos_weight = neg_num * 1. / (pos_num + neg_num)
        neg_weight = 1. - pos_weight

        mse_loss = F.smooth_l1_loss(gaussian_map, gaussian_gt, reduce='none')
        weighted_mse_loss = mse_loss * (text_map * pos_weight + bg_map * neg_weight) * training_mask
        center_region_loss = torch.sum(
            center_gt * mse_loss * training_mask) / center_gt.sum() if center_gt.sum() > 0 else 0
        text_region_loss = torch.sum(text_gt * mse_loss * training_mask) / text_map.sum() if text_map.sum() > 0 else 0
        return weighted_mse_loss.mean(), text_region_loss, center_region_loss

# def gaussian_map_loss(gaussian_map, gt_texts, training_masks, criterion, text_thres, center_thres):
#     weighted_mse_loss, mse_region_loss, loss_center = weighted_regression(gaussian_map, gt_texts, training_masks)
#
#     center_gt = torch.where(gt_texts > center_thres, gt_texts, torch.zeros_like(gt_texts))
#     region_gt = torch.where(gt_texts > text_thres, gt_texts, torch.zeros_like(gt_texts))
#
#     # loss for region_map
#     select_masks = ohem_batch(torch.sigmoid(gaussian_map), region_gt, training_masks).cuda()
#     loss_region_dice = criterion(gaussian_map, region_gt, select_masks)
#     loss = loss_center + weighted_mse_loss + mse_region_loss + loss_region_dice
#     return loss, select_masks
