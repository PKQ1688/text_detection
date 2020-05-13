import os
import shutil
from ocr_evaluation import TL_iou, rrc_evaluation_funcs
import sys


def compute_hmean(submit_file_path):
    # print('text_location <==> Evaluation <==> Compute Hmean <==> Begin')

    # basename = os.path.basename(submit_file_path)
    # assert basename == 'submit.zip', 'There is no submit.zip'
    #
    # dirname = os.path.dirname(submit_file_path)
    # gt_file_path = os.path.join(dirname, 'gt.zip')
    # assert os.path.isfile(gt_file_path), 'There is no gt.zip'
    #
    # log_file_path = os.path.join(dirname, 'log_epoch_hmean.txt')
    # if not os.path.isfile(log_file_path):
    #     os.mknod(log_file_path)
    #
    # dirname = "/home/shizai/data2/ocr_data/icdar2015/test/"
    dirname = submit_file_path
    result_dir_path = os.path.join(dirname, 'result')
    try:
        shutil.rmtree(result_dir_path)
    except:
        pass
    os.mkdir(result_dir_path)
    # gt_file = "/home/shizai/data2/ocr_data/icdar2015/test/gt.zip"
    gt_file = os.path.join(submit_file_path, 'gt.zip')
    submit_file = os.path.join(submit_file_path, 'submit.zip')

    # submit_file = "/home/shizai/data2/ocr_data/icdar2015/test/submit.zip"

    resDict = rrc_evaluation_funcs.main_evaluation({'g': gt_file, 's': submit_file, 'o': result_dir_path},
                                                   TL_iou.default_evaluation_params, TL_iou.validate_data,
                                                   TL_iou.evaluate_method)

    # print(resDict)
    recall = resDict['method']['recall']

    precision = resDict['method']['precision']

    hmean = resDict['method']['hmean']

    print('text_location <==> Evaluation <==> Precision:{:.2f} Recall:{:.2f} Hmean{:.2f} <==> Done'.format(precision,
                                                                                                           recall,
                                                                                                           hmean))

    # with open(log_file_path, 'a') as f:
    #     f.write(
    #         'text_location <==> Evaluation <==> Precision:{:.2f} Recall:{:.2f} Hmean{:.2f} <==> Done\n'.format(precision, recall,
    #                                                                                                   hmean))

    return recall, precision, hmean


if __name__ == '__main__':
    submit_file_path = '/home/shizai/data2/ocr_data/icdar2015/test/submit.zip'
    # submit_file_path = sys.argv[1]
    recall, precision, hmean = compute_hmean(submit_file_path)
