from ultralytics.models.yolo.obb import OBBValidator
from ultralytics import YOLO
import time

# 记录开始时间
start_time = time.time()

model = YOLO('/ultralytics-main/DOTAv1/runs/obb/train8/weights/best.pt')

model.val(data="/yes/ultralytics-main/cfg/datasets/yolov8n-obb_DOTAv1.yaml",
        split="test",
        conf= 0.25,
        iou=0.45, 
        batch=4,
        half=True,
        save_json=True)

# 记录结束时间
end_time = time.time()

# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.2f} 秒")