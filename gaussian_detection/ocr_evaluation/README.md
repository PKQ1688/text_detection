# ocr 评测工具使用指南

- 命名gt的文件名为gt_img_1.txt,命名submit为res_img_1.txt形式

- 将gt和需要测评的结果处理成相同的实例格式，并放在同一路径

- 修改run.sh中的文件路径，为your_dir/submit.zip

- 在命令行输入```sh run.sh```

- ps:如果有缺少的库，直接使用pip安装。python3使用安装Polygon3，暂时还支持python2
