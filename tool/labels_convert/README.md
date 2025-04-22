# 一、dior2yolo.py
## （一）useage
`python tool/labels_convert/DIOR-R/dior2yolo.py \
--xml_label_path {/ultralytics-main/Datasets/DIOR-R/Annotations/Oriented Bounding Boxes} \
--yolo_label_path {/yes/ultralytics-main/Datasets/DIOR-R/yolo-DIOR/labels}`
## （二）note
在运行命令行前，yolo-DIOR文件夹的格式如下：
The directory structure assumed for the DIOR-R dataset:

    - DIOR-R
        ├─ images
        │   ├─ trainval
        │   └─ val
且images文件夹下已经包含DIOR-R数据集的图像。

# 二、hrsc2yolo.py
## （一）useage
`python tool/labels_convert/HRSC2016/hrsc2yolo.py \
--xml_label_path {/ultralytics-main/Datasets/HRSC2016/HRSC2016} \
--yolo_label_path {/ultralytics-main/Datasets/HRSC2016/yolo_HRSC2016}`
## （二）note
在运行命令行前，yolo_HRSC2016文件夹的格式如下：
The directory structure assumed for the HRSC2016 dataset:

    - HRSC2016
        ├─ images
        │   ├─ trainval
        │   └─ val
且images文件夹下已经包含HRSC2016数据集的图像。

# 三、dota2yolo.py
## （一）usage
`python tool/labels_convert/DOTAv1/dota2yolo.py \
--DOTAv1_path {/ultralytics-main/Datasets/DOTAv1/yolo_DOTAv1} \
--crop_size 1024 \
--rates 0.5 1.0 1.5 \
--gap 500`
## （二）note
在运行命令行前，yolo_DOTAv1文件夹的格式如下：
The directory structure assumed for the DOTAv1 dataset:

    - DOTAv1
        ├─ images
        │   ├─ trainval
        │   └─ test
        ├─ labels
        │   └─ trainval_original
trainval_original是DOTA格式的标注文件。

# 四、fair1m2yolo.py
## （一）usage
`python tool/labels_convert/FAIR1M-1.0/fair1m2yolo.py \
--FAIR1M_path {/ultralytics-main/Datasets/FAIR1M1.0/yolo_FAIR1M1.0} \
--crop_size 1024 \
--rates 0.5 1.0 1.5 \
--gap 500`
## （二）note
在运行命令行前，yolo_FAIR1M文件夹的格式如下：
The directory structure assumed for the FAIR1M dataset:

    - FAIR1M
        ├─ images
        │   ├─ train
        │   └─ test
        ├─ labels
        │   └─ train_xml
train_xml是FAIR1M的xml标注文件。

# 五、yolo2fair1m.py
## （一）usage
`python tool/labels_convert/FAIR1M-1.0/yolo2fair1m.py \
--merged_dir {/ultralytics-main/FAIR1M1.0/runs/obb/val15/predictions_merged_txt} \`
