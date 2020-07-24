# -*- coding:utf-8 -*-
# @author :adolf
from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to obtain the shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


if __name__ == '__main__':
    from typing import Dict

    input = Dict[str, ShapeSpec]
    print(input)
    Sh = ShapeSpec(channels=3, height=23, width=12, stride=3)
    print(Sh)
