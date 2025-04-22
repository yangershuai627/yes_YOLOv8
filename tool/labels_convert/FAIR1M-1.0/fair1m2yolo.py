import argparse
import os
import xml.etree.ElementTree as ET
from ultralytics.data.converter import convert_dota_to_yolo_obb

from pathlib import Path
import itertools

from math import ceil

import cv2

from ultralytics.utils import TQDM

import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils.checks import check_requirements

check_requirements("shapely")
from shapely.geometry import Polygon
from glob import glob

def bbox_iof(polygon1, bbox2, eps=1e-6):
    """
    Calculate Intersection over Foreground (IoF) between polygons and bounding boxes.

    Args:
        polygon1 (np.ndarray): Polygon coordinates, shape (n, 8).
        bbox2 (np.ndarray): Bounding boxes, shape (n, 4).
        eps (float, optional): Small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        (np.ndarray): IoF scores, shape (n, 1) or (n, m) if bbox2 is (m, 4).

    Note:
        Polygon format: [x1, y1, x2, y2, x3, y3, x4, y4].
        Bounding box format: [x_min, y_min, x_max, y_max].
    """
    polygon1 = polygon1.reshape(-1, 4, 2)
    lt_point = np.min(polygon1, axis=-2)  # left-top
    rb_point = np.max(polygon1, axis=-2)  # right-bottom
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    left, top, right, bottom = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)

    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def load_yolo_dota(data_root, split="train"):
    """
    Load DOTA dataset.

    Args:
        data_root (str): Data root.
        split (str): The split data set, could be train or val.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train"}, f"Split must be 'train', not {split}."
    # original
    # assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos


def get_windows(im_size, crop_sizes=(1024,), gaps=(200,), im_rate_thr=0.6, eps=0.01):
    """
    Get the coordinates of windows.

    Args:
        im_size (tuple): Original image size, (h, w).
        crop_sizes (List(int)): Crop size of windows.
        gaps (List(int)): Gap between crops.
        im_rate_thr (float): Threshold of windows areas divided by image ares.
        eps (float): Epsilon value for math operations.
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno, windows, iof_thr=0.7):
    """Get objects for each window."""
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        # Unnormalized and misaligned coordinates
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns


def crop_and_save(anno, windows, window_objs, im_dir, lb_dir, allow_background_images=True):
    """
    Crop images and save new labels.

    Args:
        anno (dict): Annotation dict, including `filepath`, `label`, `ori_size` as its keys.
        windows (list): A list of windows coordinates.
        window_objs (list): A list of labels inside each window.
        im_dir (str): The output directory path of images.
        lb_dir (str): The output directory path of labels.
        allow_background_images (bool): Whether to include background images without labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    # # 去除不包含目标的Patch
    # allow_background_images = False
    
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        label = window_objs[i]
        if len(label) or allow_background_images:
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.tif"), patch_im)
        if len(label):
            label[:, 1::2] -= x_start
            label[:, 2::2] -= y_start
            label[:, 1::2] /= pw
            label[:, 2::2] /= ph

            # ✨ 添加裁剪限制：坐标必须在 [0, 1] 范围内
            # 对于DOTAv1 FAIR1M-1.0数据集，裁图时对其进行坐标归一化处理，以减小误差
            label[:, 1:] = np.clip(label[:, 1:], 0, 1)

            with open(Path(lb_dir) / f"{new_name}.txt", "w") as f:
                for lb in label:
                    formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                    f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


def split_images_and_labels(data_root, save_dir, split="train", crop_sizes=(1024,), gaps=(200,)):
    """
    Split both images and labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root, split=split)
    for anno in tqdm(annos, total=len(annos), desc=split):
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows) # 将标注转换后坐标恢复为原始坐标。如果在标注转换这里对坐标归一化处理，恢复为原始坐标的时候就可能引入误差。
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))


def split_trainval(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split train and val set of DOTA.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        and the output directory structure is:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    # FAIR1M-1.0 dataset
    for split in ["train"]:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)
    # original
    # for split in ["train", "val"]:
    #     split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)


def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split test set of DOTA, labels are not included within this set.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - test
        and the output directory structure is:
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / "*"))
    for im_file in tqdm(im_files, total=len(im_files), desc="test"):
        w, h = exif_size(Image.open(im_file))
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.tif"), patch_im)


def split_dataset(FAIR1M_path, crop_size, rates, gap):
    parent = os.path.dirname(FAIR1M_path)
    split_dir = os.path.join(parent, 'yolo_FAIR1M_split')
    os.makedirs(split_dir, exist_ok=True)
    
    # split trainval set, with labels.
    split_trainval(
        data_root=FAIR1M_path,
        save_dir=split_dir, # allow_background_images is False
        rates=rates,  # multiscale
        crop_size=crop_size,
        gap=gap
    )

    # split test set, without labels.
    split_test(
        data_root=FAIR1M_path,
        save_dir=split_dir,
        rates=rates,  # multiscale
        crop_size=crop_size,
        gap=gap
    )


def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # Class names to indices mapping
    class_mapping = {
        "Boeing737": 0,
        "Boeing747": 1,
        "Boeing777": 2,
        "Boeing787": 3,
        "C919": 4,
        "A220": 5,
        "A321": 6,
        "A330": 7,
        "A350": 8,
        "ARJ21": 9,
        "other-airplane": 10,
        "Passenger Ship": 11,
        "Motorboat": 12,
        "Fishing Boat": 13,
        "Tugboat": 14,
        "Engineering Ship": 15,
        "Liquid Cargo Ship": 16,
        "Dry Cargo Ship": 17,
        "Warship": 18,
        "other-ship": 19,
        "Small Car": 20,
        "Bus": 21,
        "Cargo Truck": 22,
        "Dump Truck": 23,
        "Van": 24,
        "Trailer": 25,
        "Tractor": 26,
        "Excavator": 27,
        "Truck Tractor": 28,
        "other-vehicle": 29,
        "Basketball Court": 30,
        "Tennis Court": 31,
        "Football Field": 32,
        "Baseball Field": 33,
        "Intersection": 34,
        "Roundabout": 35,
        "Bridge": 36,
}        

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = " ".join(parts[8:]) # 将像Liquid Cargo Ship这样的类名合并为一个整体
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                # # 限制归一化的坐标范围在 [0, 1]
                # # 处理DOTAv1 FAIR1M数据集时首先进行标注转换，在进行裁图。标注转换时不对坐标归一化，裁图时归一化，以减小数据预处理误差
                # normalized_coords = [max(0, min(1, coord)) for coord in normalized_coords]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    # # FAIR1M dataset 的test只有图像没有标签，因此我们只处理train
    for phase in ["train"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".tif":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def label_convert(FAIR1M_path):
        convert_dota_to_yolo_obb(FAIR1M_path)
        print('标注转换完成')


def xml2dota_txt(FAIR1M_path):
    xml_dir = os.path.join(FAIR1M_path, 'labels', 'train_xml')
    dota_original = os.path.join(FAIR1M_path, 'labels', 'train_original')
    # 确保输出文件夹存在
    os.makedirs(dota_original, exist_ok=True)
    
    # 遍历输入文件夹中的所有XML文件
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_file_path = os.path.join(xml_dir, xml_file)
            # 读取XML文件
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            # 构建输出txt文件路径
            txt_file = os.path.join(dota_original, os.path.splitext(xml_file)[0] + '.txt')
            # 打开txt文件进行写入
            with open(txt_file, 'w') as f:
                # 循环遍历每个object节点
                for obj in root.findall('.//object'):
                    # 提取需要的信息
                    points = obj.find('.//points')
                    point_list = points.findall('.//point')
                    if len(point_list) >= 4:
                        p1 = point_list[0].text
                        p1 = p1.split(',')
                        x1 = float(p1[0])
                        y1 = float(p1[1])

                        p2 = point_list[1].text
                        p2 = p2.split(',')
                        x2 = float(p2[0])
                        y2 = float(p2[1])

                        p3 = point_list[2].text
                        p3 = p3.split(',')
                        x3 = float(p3[0])
                        y3 = float(p3[1])

                        p4 = point_list[3].text
                        p4 = p4.split(',')
                        x4 = float(p4[0])
                        y4 = float(p4[1])
                    
                    name = obj.find('.//possibleresult/name').text

                    # 写入到txt文件
                    line = f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {name}\n"
                    f.write(line)
    print("xml2dota_original批量转换完成。")


def main(): 
    parser = argparse.ArgumentParser(description="Convert FAIR1M XML to YOLO format")
    parser.add_argument('--FAIR1M_path', 
                        type=str, 
                        required=True, 
                        help='Path to the input DOTAv1 folder')
    
    parser.add_argument('--crop_size', 
                    type=int, 
                    required=True, 
                    default=1024,
                    help='crop size of the image')
    parser.add_argument('--rates', 
                type=float, 
                required=True,
                nargs='+', # 接受多个值 
                default=[0.5, 1.0, 1.5],
                help='list of scales to crop the image')
    parser.add_argument('--gap', 
            type=int, 
            required=True, 
            default=500,
            help='gap between crops')
    args = parser.parse_args()
    args.rates = [float(i) for i in args.rates]
    # # 设置默认参数（用于调试模式）
    # if not args.FAIR1M_path:
    #     args.FAIR1M_path = "/disk2/xiexingxing/home/yes/ultralytics-main/Datasets/FAIR1M1.0/yolo_FAIR1M1.0"

    xml2dota_txt(args.FAIR1M_path)
    label_convert(args.FAIR1M_path)
    split_dataset(args.FAIR1M_path, args.crop_size, args.rates, args.gap)    


if __name__ == "__main__":
    main()