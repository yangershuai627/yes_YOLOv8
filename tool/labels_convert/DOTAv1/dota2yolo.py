# DOTA-v1数据集的处理思路是先对其进行多尺度裁图，后在patch坐标系下进行标注转换
# note:
# 1. 在裁图时不进行坐标归一化处理，最后在标注转换时进行归一化处理，这样可以减小误差
# 2. crop_size=1024, rates=[0.5, 1.0, 1.5]，gap=500
import argparse
from ultralytics.data.split_dota import split_test, split_trainval
import os
from ultralytics.data.converter import convert_dota_to_yolo_obb

def split_dataset(DOTAv1_path, crop_size, rates, gap):
    parent = os.path.dirname(DOTAv1_path)
    split_dir = os.path.join(parent, 'DOTAv1_yolo_split')
    
    # split trainval set, with labels.
    split_trainval(
        data_root=DOTAv1_path,
        save_dir=split_dir, # allow_background_images is False
        rates=rates,  # multiscale
        crop_size=crop_size,
        gap=gap
    )

    # split test set, without labels.
    split_test(
        data_root=DOTAv1_path,
        save_dir=split_dir,
        rates=rates,  # multiscale
        crop_size=crop_size,
        gap=gap
    )


def label_convert(yolo_DOTAv1_path):
    convert_dota_to_yolo_obb(yolo_DOTAv1_path)


def main():
    parser = argparse.ArgumentParser(description="Convert DOTA-v1 annotations to YOLO format")

    parser.add_argument('--DOTAv1_path', 
                        type=str, 
                        required=True, 
                        help='Path to the input DOTAv1 folder')
    parser.add_argument('--yolo_DOTAv1_path', 
                        type=str, 
                        required=True, 
                        help='Path to the output yolo-DOTAv1 folder')
    
    parser.add_argument('--crop_size', 
                    type=int, 
                    required=True, 
                    default=1024,
                    help='crop size of the image')
    parser.add_argument('--rates', 
                type=list, 
                required=True, 
                default=[0.5, 1.0, 1.5],
                help='list of scales to crop the image')
    parser.add_argument('--gap', 
            type=int, 
            required=True, 
            default=500,
            help='gap between crops')

    args = parser.parse_args()

    split_dataset(args.DOTAv1_path, args.crop_size, args.rates, args.gap)
    label_convert(args.yolo_DOTAv1_path)


if __name__ == '__main__':
    main()
