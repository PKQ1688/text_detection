# -*- coding:utf-8 -*-
# @author :adolf
from tools.predict import OnePredict
import yaml
import os
from tqdm import tqdm

from ocr_evaluation.hmean import compute_hmean


class EvalChinaLife(object):
    def __init__(self, eval_path, use_model=None):
        self.eval_path = eval_path
        self.img_path = os.path.join(self.eval_path, 'imgs')
        self.img_list = os.listdir(self.img_path)
        with open('config/db_resnet50.yaml', 'r') as fp:
            config = yaml.load(fp.read(), Loader=yaml.FullLoader)
        self.img_predict = OnePredict(configs=config, use_model=use_model)

    def out_file(self, out_path):
        i = 0
        for img_name in self.img_list:
            is_vis = False
            # if i % 200 == 0:
            #     is_vis = True
            # else:
            #     is_vis = False
            _ = self.img_predict.inference(
                img_path=os.path.join(self.img_path, img_name),
                is_visualize=is_vis, is_format_output=True, out_path=out_path)
            i += 1

    def main(self):
        out_path = os.path.join(self.eval_path, 'submit')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        self.out_file(out_path)
        os.system('cd ' + out_path + ';zip -q -r submit.zip *;mv submit.zip ..')
        # shutil.move('submit.zip', os.path.join(eval_path, 'submit.zip'))
        recall, precision, hmean = compute_hmean(self.eval_path)
        return recall, precision, hmean


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    eval_path = '/home/shizai/data2/ocr_data/china_life_test_data/'
    _, _, hmean = EvalChinaLife(eval_path, use_model='weights/DB_final.pth').main()
