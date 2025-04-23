import os
import xml.etree.ElementTree as ET
import argparse
import numpy as np
from ultralytics.data.converter import convert_dota_to_yolo_obb

def xml2txt(xml_label_path):

    for phase in ["Trainval", "Test"]:
        xml_label_dir = os.path.join(xml_label_path, phase, 'Annotations')
        
        txt_label_dir = os.path.join(xml_label_path, phase, 'Annotations_txt')
        os.makedirs(txt_label_dir, exist_ok=True)

        for xml_file in os.listdir(xml_label_dir):
            if xml_file.endswith(".xml"):
                xml_path = os.path.join(xml_label_dir, xml_file)
                # 读取XML文件
                tree = ET.parse(xml_path)
                root = tree.getroot()
                # 构建输出txt文件路径
                txt_file = os.path.join(txt_label_dir, os.path.splitext(xml_file)[0] + '.txt')
                # 打开txt文件进行写入
                with open(txt_file, 'w') as f:
                    # 循环遍历每个object节点
                    for obj in root.findall('.//HRSC_Object'):
                        # 提取需要的信息
                        x_c = obj.find('.//mbox_cx').text
                        y_c = obj.find('.//mbox_cy').text
                        w = obj.find('.//mbox_w').text
                        h = obj.find('.//mbox_h').text
                        angle = obj.find('.//mbox_ang').text

                        # 写入到txt文件
                        line = f"{x_c} {y_c} {w} {h} {angle} {'ship'}\n"
                        f.write(line)
        print(f"{phase}批量转换完成。")

def obb2poly_le90_CW(float_parts):
    xc, yc, w, h, theta = float_parts

    Cos, Sin = np.cos(theta), np.sin(theta)
    CW_matrix = np.array([[Cos, -Sin], [Sin, Cos]])

    point1 = CW_matrix @ np.array([-w / 2, -h / 2]) + np.array([xc, yc])
    point2 = CW_matrix @ np.array([w / 2, -h / 2]) + np.array([xc, yc])
    point3 = CW_matrix @ np.array([w / 2, h / 2]) + np.array([xc, yc])
    point4 = CW_matrix @ np.array([-w / 2, h / 2]) + np.array([xc, yc])

    return [point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], point4[0], point4[1]]


def xywh2xyxyxyxy(xml_label_path):
    parent_path = os.path.dirname(xml_label_path)
    yolo_HRSC2016_path = os.path.join(parent_path, 'yolo_HRSC2016')

    for phase in ["Trainval", "Test"]:
        txt_label_dir = os.path.join(xml_label_path, phase, 'Annotations_txt')
        phase_lower = phase.lower()
        poly_label_dir = os.path.join(yolo_HRSC2016_path, 'labels', f'{phase_lower}_original')
        os.makedirs(poly_label_dir, exist_ok=True)

        for txt_file in os.listdir(txt_label_dir):
            if txt_file.endswith(".txt"):
                txt_path = os.path.join(txt_label_dir, txt_file)
                # 读取txt文件
                with open(txt_path, 'r') as f:
                    converted = []
                    for line in f:
                        # 分割字符串，得到每一列的数据
                        parts = line.split()
                        # 将数字部分（前5个元素）转换为浮点数
                        float_parts = [float(x) for x in parts[:-1]]  # 排除最后的 "ship" 字符串
                        # 将obb格式转换为poly格式
                        float_parts = obb2poly_le90_CW(float_parts)
                        # 将转换后的浮点数部分与 "ship" 结合
                        converted.append(float_parts + [parts[-1]])
                    
            # 构建输出txt文件路径
            poly_label_file = os.path.join(poly_label_dir, os.path.splitext(txt_file)[0] + '.txt')
            # 打开txt文件进行写入
            with open(poly_label_file, 'w') as t:
                for line in converted:
                    line = ' '.join([str(x) for x in line])
                    t.write(line + '\n')

    print("obb2poly格式转换完成。")


def yolo_label(yolo_label_path):
    convert_dota_to_yolo_obb(yolo_label_path)



def main():
    parser = argparse.ArgumentParser(description="Convert HRSC2016 annotations to YOLO format")

    parser.add_argument('--xml_label_path', 
                        type=str, 
                        required=True, 
                        help='Path to the input original label folder')
    parser.add_argument('--yolo_label_path',
                        type=str, 
                        required=True, 
                        help='Path to the output yolo label folder')
    
    args = parser.parse_args()

    xml2txt(args.xml_label_path)
    xywh2xyxyxyxy(args.xml_label_path)
    yolo_label(args.yolo_label_path)


if __name__ == "__main__":
    main()
