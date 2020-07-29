# -*- coding:utf-8 -*-
# @author :adolf
import torch
from torch import nn
from Model.Loss import IOULoss, sigmoid_focal_loss_jit
import torch.nn.functional as F

from Model.layers import cat, ml_nms
from Model.structures import Instances, Boxes
from utils.comm import get_world_size, reduce_sum

INF = 100000000


def compute_ctrness_targets(reg_targets):
    """

    :param reg_targets:
    :return:
    """
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class FCOSLoss(nn.Module):
    def __init__(self, cfg):
        super(FCOSLoss, self).__init__()
        self.focal_loss_alpha = cfg.LOSS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.LOSS.LOSS_GAMMA
        self.center_sample = cfg.LOSS.CENTER_SAMPLE

        self.loc_loss_func = IOULoss(cfg.LOSS.LOC_LOSS_TYPE)

    @staticmethod
    def _transpose(training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] /= float(self.strides[l])

        return training_targets

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds
        }

    def forward(self, pred, gt):
        logits_pred = pred["logits_pred"]
        reg_pred = pred["reg_pred"]
        ctrness_pred = pred["ctrness_pred"]
        top_feats = pred["top_feats"]
        # bbox_towers = pred["bbox_towers"]
        locations = pred["locations"]

        training_targets = self._get_ground_truth(locations, gt)

        instances = Instances((0, 0))
        instances.labels = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["labels"]
        ], dim=0)
        instances.gt_inds = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["target_inds"]
        ], dim=0)
        instances.im_inds = cat([
            x.reshape(-1) for x in training_targets["im_inds"]
        ], dim=0)
        instances.reg_targets = cat([
            # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            x.reshape(-1, 4) for x in training_targets["reg_targets"]
        ], dim=0, )
        instances.locations = cat([
            x.reshape(-1, 2) for x in training_targets["locations"]
        ], dim=0)
        instances.fpn_levels = cat([
            x.reshape(-1) for x in training_targets["fpn_levels"]
        ], dim=0)

        instances.logits_pred = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred
        ], dim=0, )
        instances.reg_pred = cat([
            # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred
        ], dim=0, )
        instances.ctrness_pred = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness_pred
        ], dim=0, )

        if len(top_feats) > 0:
            instances.top_feats = cat([
                # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in top_feats
            ], dim=0, )

        return self.fcos_losses(instances)

    def fcos_losses(self, instances):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        if pos_inds.numel() > 0:
            reg_loss = self.loc_loss_func(
                instances.reg_pred,
                instances.reg_targets,
                ctrness_targets
            ) / loss_denorm

            ctrness_loss = F.binary_cross_entropy_with_logits(
                instances.ctrness_pred,
                ctrness_targets,
                reduction="sum"
            ) / num_pos_avg
        else:
            reg_loss = instances.reg_pred.sum() * 0
            ctrness_loss = instances.ctrness_pred.sum() * 0

        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_ctr": ctrness_loss
        }
        extras = {
            "instances": instances,
            "loss_denorm": loss_denorm
        }
        return extras, losses
