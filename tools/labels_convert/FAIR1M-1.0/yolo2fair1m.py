import argparse
import os
import xml.etree.ElementTree as ET
import re
from xml.dom import minidom

def format_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed_str = minidom.parseString(xml_str)
    formatted_str = parsed_str.toprettyxml(indent="    ")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_str)


def create_xml(output_dir, xml_name, img_name, objects_data):
        
    annotation = ET.Element('annotation')

    # 创建 <source> 节点
    source = ET.SubElement(annotation, 'source')
    filename_element = ET.SubElement(source, 'filename')
    filename_element.text = img_name
    origin_element = ET.SubElement(source, 'origin')
    origin_element.text = 'GF2/GF3'

    # 创建 <research> 节点
    research = ET.SubElement(annotation, 'research')
    version_element = ET.SubElement(research, 'version')
    version_element.text = '1.0'
    provider_element = ET.SubElement(research, 'provider')
    provider_element.text = 'FAIR1M'
    author_element = ET.SubElement(research, 'author')
    author_element.text = 'YES'
    pluginname_element = ET.SubElement(research, 'pluginname')
    pluginname_element.text = 'FAIR1M'
    pluginclass_element = ET.SubElement(research, 'pluginclass')
    pluginclass_element.text = 'object detection'
    time_element = ET.SubElement(research, 'time')
    time_element.text = '2025-01-11'

    # 创建新的 <object> 节点并添加
    objects = ET.SubElement(annotation, 'objects')

    for obj_data in objects_data:
        x1, y1, x2, y2, x3, y3, x4, y4, class_name, conf = obj_data

        # 创建 <object> 节点
        object = ET.SubElement(objects, 'object')
        coordinate_element = ET.SubElement(object, 'coordinate')
        coordinate_element.text = 'pixel'
        type_element = ET.SubElement(object, 'type')
        type_element.text = 'rectangle'
        description_element = ET.SubElement(object, 'description')
        description_element.text = 'None'
        possibleresult_element = ET.SubElement(object, 'possibleresult')

        # 需要更改为文件名对应的类别
        name = ET.SubElement(possibleresult_element, 'name')
        class_name = re.sub(r'(?<!other)-', ' ', class_name)
        name.text = class_name

        # 需要更改为文件名对应的置信度
        probability_element = ET.SubElement(possibleresult_element, 'probability')
        probability_element.text = str(conf)

        # 添加标注框的四个点
        points = ET.SubElement(object, 'points')
        point1 = ET.SubElement(points, 'point')
        point1.text = f"{x1},{y1}"
        point2 = ET.SubElement(points, 'point')
        point2.text = f"{x2},{y2}"
        point3 = ET.SubElement(points, 'point')
        point3.text = f"{x3},{y3}"
        point4 = ET.SubElement(points, 'point')
        point4.text = f"{x4},{y4}"
        point5 = ET.SubElement(points, 'point')
        point5.text = f"{x1},{y1}"  # 闭合点

    # 创建或更新 XML 树并保存为文件
    tree = ET.ElementTree(annotation)
    xml_path = os.path.join(output_dir, xml_name)
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    format_xml(xml_path)
    print("XML 文件已保存到 {}".format(xml_name))


def create_image_name_dict(merged_dir):
    image_name_dict = {}

    # 遍历文件夹中的所有 .txt 文件
    for txt_file in os.listdir(merged_dir):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(merged_dir, txt_file)
            class_name = txt_file.split('_')[1].split('.')[0]  # 提取类别名

            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    parts = line.split(' ')
                    img_name = parts[0]  # 图片名
                    conf = parts[1]  # 置信度
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[2:10])  # 四个坐标值（转为浮点数）
                    
                    # 如果该图片没有在字典中，初始化字典结构
                    if img_name not in image_name_dict:
                        image_name_dict[img_name] = []

                    # 将该预测框添加到对应类别的列表中
                    image_name_dict[img_name].append([x1, y1, x2, y2, x3, y3, x4, y4, class_name, conf])
    
    return image_name_dict


def fair1m2yolo(merged_dir):
    image_name_dict = create_image_name_dict(merged_dir)
    parent = os.path.dirname(merged_dir)
    xml_dir_path = os.path.join(parent, 'predictions_merged_xml')
    for img_name, objects in image_name_dict.items():
        # 创建 XML 文件
        objects_data = objects  # 将字典中的值复制给 objects_data
        
        xml_file_name = img_name + '.xml'
        create_xml(xml_dir_path, xml_file_name, img_name, objects_data)


def main():
    parser = argparse.ArgumentParser(description="将合图后得到的.txt转换为fair1m的.xml格式")

    parser.add_argument('--merged_dir',
                        type=str, 
                        required=True,
                        default='/disk2/xiexingxing/home/yes/ultralytics-main/FAIR1M1.0/runs/obb/val15/predictions_merged_txt',
                        help='merged_dir')

    args = parser.parse_args()

    fair1m2yolo(args.merged_dir)


if __name__ == "__main__":
    main()