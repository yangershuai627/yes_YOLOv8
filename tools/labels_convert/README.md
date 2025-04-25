# 一、dior2yolo.py
## （一）useage
`python tools/labels_convert/DIOR-R/dior2yolo.py
--xml_label_path {/ultralytics-main/Datasets/DIOR-R/Annotations/Oriented Bounding Boxes} \
--yolo_label_path {/yes/ultralytics-main/Datasets/DIOR-R/yolo_DIOR/labels}`
{}内的路径替换为你的实际路径，运行CLI要去掉{}，以下同理。
## （二）note
在运行命令行前，yolo-DIOR文件夹的格式如下：
The directory structure assumed for the DIOR-R dataset:

    - yolo_DIOR-R
        ├─ images
        │   ├─ trainval
        │   └─ test
且images文件夹下已经包含DIOR-R数据集的图像。

# 二、hrsc2yolo.py
## （一）useage
`python tools/labels_convert/HRSC2016/hrsc2yolo.py \
--xml_label_path {/ultralytics-main/Datasets/HRSC2016/HRSC2016} \
--yolo_label_path {/ultralytics-main/Datasets/HRSC2016/yolo_HRSC2016}`
## （二）note
在运行命令行前，yolo_HRSC2016文件夹的格式如下：
The directory structure assumed for the HRSC2016 dataset:

    - yolo_HRSC2016
        ├─ images
        │   ├─ trainval
        │   └─ val
且images文件夹下已经包含HRSC2016数据集的图像。

# 三、dota2yolo.py
## （一）usage
`python tools/labels_convert/DOTAv1/dota2yolo.py \
--DOTAv1_path {/ultralytics-main/Datasets/DOTAv1/yolo_DOTAv1} \
--crop_size 1024 \
--rates 0.5 1.0 1.5 \
--gap 500`
## （二）note
在运行命令行前，yolo_DOTAv1文件夹的格式如下：
The directory structure assumed for the DOTAv1 dataset:

    - yolo_DOTAv1
        ├─ images
        │   ├─ trainval
        │   └─ test
        ├─ labels
        │   └─ trainval_original
trainval_original是DOTA格式的标注文件。
## （三）DOTAv1数据集train val test的划分
DOTAv1数据集一共包含2806张图像。其中train有1411张，val有458张，test有937张。

# 四、fair1m2yolo.py
## （一）usage
`python tools/labels_convert/FAIR1M-1.0/fair1m2yolo.py \
--FAIR1M_path {/ultralytics-main/Datasets/FAIR1M1.0/yolo_FAIR1M1.0} \
--crop_size 1024 \
--rates 0.5 1.0 1.5 \
--gap 500`
## （二）note
在运行命令行前，yolo_FAIR1M文件夹的格式如下：
The directory structure assumed for the FAIR1M dataset:

    - yolo_FAIR1M
        ├─ images
        │   ├─ train
        │   └─ test
        ├─ labels
        │   └─ train_xml
train_xml是FAIR1M的xml标注文件。

# 五、yolo2fair1m.py
## （一）usage
`python tools/labels_convert/FAIR1M-1.0/yolo2fair1m.py \
--merged_dir {/ultralytics-main/FAIR1M1.0/runs/obb/val15/predictions_merged_txt} \`
