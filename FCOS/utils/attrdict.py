# -*- coding:utf-8 -*-
# @author :adolf

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == '__main__':
    a = {"a": 1, "b": 2, "c": 3}
    cfg = AttrDict(a)
    print(cfg.a)
    print(cfg.keys())