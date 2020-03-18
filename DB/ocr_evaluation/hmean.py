import os
import shutil
import TL_iou, rrc_evaluation_funcs
import sys


def compute_hmean(submit_file_path):
    print('text_location <==> Evaluation <==> Compute Hmean <==> Begin')

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
    dirname = "/home/shizai/adolf/ai+rpa/ocr/ocr_pra/maskrcnn/dataset/"
    result_dir_path = os.path.join(dirname, 'result')
    try:
        shutil.rmtree(result_dir_path)
    except:
        pass
    os.mkdir(result_dir_path)
    gt_file_path = "/home/shizai//datadisk4/SynthText/miao/gt.zip"
    # submit_file_path = "/home/shizai/adolf/ai+rpa/ocr/ocr_use/PMTD/submit.zip"
    submit_file_path = "/home/shizai/adolf/ai+rpa/ocr/ocr_use/keras-ctpn/submit.zip"
    # submit_file_path = "/home/shizai/adolf/ai+rpa/ocr/ocr_pra/mytest/miao_submit.zip"

    resDict = rrc_evaluation_funcs.main_evaluation({'g': gt_file_path, 's': submit_file_path, 'o': result_dir_path},
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

    return hmean


if __name__ == '__main__':
    # submit_file_path = '/home/shizai/adolf/ai+rpa/ocr/git_test/EAST/result/submit.zip'
    submit_file_path = sys.argv[1]
    hmean = compute_hmean(submit_file_path)
