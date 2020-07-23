from typing import List
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms


def batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    
    Args:
        boxlist (detectron2.structures.Boxes): 
        nms_thresh (float): 
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str): 
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist
