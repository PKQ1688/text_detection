# -*- coding:utf-8 -*-
# @author :adolf
import yaml

pre_processes = [{'type': 'IaaAugment', 'args': [{'type': 'Fliplr', 'args': {'p': 0.5}},
                                                 {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                                                 {'type': 'Resize', 'args': {'size': [0.5, 3]}}]},
                 {'type': 'EastRandomCropData', 'args': {'size': [640, 640], 'max_tries': 50, 'keep_ratio': True}},
                 {'type': 'MakeBorderMap', 'args': {'shrink_ratio': 0.4}},
                 {'type': 'MakeShrinkMap', 'args': {'shrink_ratio': 0.4, 'min_text_size': 8}}]

# fp = open('tmp.yaml', 'w')
# fp.write(yaml.dump(pre_processes))
# fp.close()

fp = open('tmp.yaml', 'r')
st = fp.read()
fp.close()

dd = yaml.load(st, Loader=yaml.FullLoader)
print(dd)

fp = open('db_resnet50.yaml', 'r')
st = fp.read()
fp.close()

dd = yaml.load(st, Loader=yaml.FullLoader)
print(dd)