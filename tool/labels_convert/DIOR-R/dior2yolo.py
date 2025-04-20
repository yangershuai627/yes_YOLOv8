import os
import xml.etree.ElementTree as ET
import argparse
import shutil
from ultralytics.data.converter import convert_dota_to_yolo_obb

def xml2txt(xml_label_path):

    parent_path = os.path.dirname(xml_label_path)
    txt_label_path = os.path.join(parent_path, 'yolo_Oriented Bounding Boxes')
    os.makedirs(txt_label_path, exist_ok=True)

    # 遍历输入文件夹中的所有XML文件
    for xml_file in os.listdir(xml_label_path):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_label_path, xml_file)
            # 读取XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 构建输出txt文件路径
            txt_file = os.path.join(txt_label_path, os.path.splitext(xml_file)[0] + '.txt')
            # 打开txt文件进行写入
            with open(txt_file, 'w') as f:
                # 循环遍历每个object节点
                for obj in root.findall('.//object'):
                    # 提取需要的信息
                    x_left_top = obj.find('.//x_left_top').text
                    y_left_top = obj.find('.//y_left_top').text
                    x_right_top = obj.find('.//x_right_top').text
                    y_right_top = obj.find('.//y_right_top').text
                    x_right_bottom = obj.find('.//x_right_bottom').text
                    y_right_bottom = obj.find('.//y_right_bottom').text
                    x_left_bottom = obj.find('.//x_left_bottom').text
                    y_left_bottom = obj.find('.//y_left_bottom').text
                    name = obj.find('.//name').text
                    angle = obj.find('.//angle').text
                    # 写入到txt文件
                    line = f"{x_left_top} {y_left_top} {x_right_top} {y_right_top} {x_right_bottom} {y_right_bottom} {x_left_bottom} {y_left_bottom} {name} {angle}\n"
                    f.write(line)
    
    print("xml2txt批量转换完成。")
    return txt_label_path


def split_files(source_folder, trainval_label_path, test_label_path):

    os.makedirs(trainval_label_path, exist_ok=True)
    os.makedirs(test_label_path, exist_ok=True)

    # 获取源文件夹中的所有 .txt 文件，并按文件名排序
    txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]
    txt_files.sort()  # 按照文件名排序

    # 计算分割点，50% 的文件到 trainval_label_path，剩下的到 test_label_path
    split_point = 11725

    # 将前一半文件复制到 trainval_label_path
    for i in range(split_point):
        src_file = os.path.join(source_folder, txt_files[i])
        dest_file = os.path.join(trainval_label_path, txt_files[i])
        shutil.copy(src_file, dest_file)
        print(f"复制文件 {src_file} 到 {trainval_label_path}")

    # 将后一半文件复制到 test_label_path
    for i in range(split_point, len(txt_files)):
        src_file = os.path.join(source_folder, txt_files[i])
        dest_file = os.path.join(test_label_path, txt_files[i])
        shutil.copy(src_file, dest_file)
        print(f"复制文件 {src_file} 到 {test_label_path}")


def yolo_label(yolo_label_path):
    parent_path = os.path.dirname(yolo_label_path)
    convert_dota_to_yolo_obb(parent_path)


def main():
    parser =argparse.ArgumentParser(description="Convert DIOR-R annotations to YOLO format")

    parser.add_argument('--xml_label_path', type=str, required=True, help='Path to the input origianl label folder')
    parser.add_argument('--yolo_label_path', type=str, required=True, help='Path to the output yolo label folder')

    args = parser.parse_args()

    args.txt_label_path = xml2txt(args.xml_label_path)
    split_files(args.txt_label_path, os.path.join(args.yolo_label_path, 'trainval_original'), os.path.join(args.yolo_label_path, 'test_original'))
    yolo_label(args.yolo_label_path)


if __name__ == "__main__":
    main()
