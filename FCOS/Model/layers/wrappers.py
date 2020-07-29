# -*- coding:utf-8 -*-
# @author :adolf
from typing import List

import torch


def cat(tensors: object, dim: object = 0) -> object:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    :rtype:
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None
