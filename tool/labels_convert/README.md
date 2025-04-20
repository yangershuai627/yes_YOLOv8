# 一、dior2yolo.py
## （一）useage
python tool/labels_convert/DIOR-R/dior2yolo.py \
--xml_label_path {/ultralytics-main/Datasets/DIOR-R/Annotations/Oriented Bounding Boxes} --yolo_label_path {/yes/ultralytics-main/Datasets/DIOR-R/yolo-DIOR/labels}
## （二）note
在运行命令行前，yolo-DIOR文件夹的格式如下：
The directory structure assumed for the DOTA dataset:

    - DOTA
        ├─ images
        │   ├─ trainval
        │   └─ val
且images文件夹下已经包含DIOR-R数据集的图像。
