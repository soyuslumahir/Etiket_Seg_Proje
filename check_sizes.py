import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ETIKET_MODEL_PATH = Path(__file__).parent / "runs/segment/etiket_seg_only/weights/best.pt"
DIGIT_MODEL_PATH  = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"
TEST_DIR = Path(__file__).parent / "test_gercek_aug"

print("Modeller yukleniyor...")
etiket_model = YOLO(str(ETIKET_MODEL_PATH))
digit_model  = YOLO(str(DIGIT_MODEL_PATH))
print("Hazir.\n")

# Sadece orijinal 2 fotografi kullan
imgs = [
    TEST_DIR / "usb_frame_20260622_103722_033.jpg",
    TEST_DIR / "usb_frame_20260622_103815_642.jpg",
]

for img_path in imgs:
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    img_area = W * H

    res_etiket = etiket_model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]
    res_digit  = digit_model(img_bgr, conf=0.25, imgsz=2048, verbose=False)[0]

    # Etiket kutulari
    etiket_boxes = []
    if res_etiket.masks is not None:
        for conf, box in zip(res_etiket.boxes.conf, res_etiket.boxes.xyxy):
            b  = box.tolist()
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            area = bw * bh
            if area < img_area * 0.003 or not (0.3 <= bw/bh <= 3.0):
                continue
            etiket_boxes.append(b)

    # QR tespiti ve konumu
    qr_boxes = []
    if res_digit.masks is not None:
        for cls, box in zip(res_digit.boxes.cls, res_digit.boxes.xyxy):
            if int(cls) == 3:  # QR
                qr_boxes.append(box.tolist())

    print(f"\n{img_path.name}  ({W}x{H})")
    print(f"{'#':<4} {'Alan(px2)':<14} {'Alan(%)':<10} {'BxH':<18} {'QR var?'}")
    print("-" * 60)

    for i, b in enumerate(sorted(etiket_boxes, key=lambda x: x[0]), 1):
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        area = bw * bh
        pct  = area / img_area * 100
        # QR bu etikete ait mi?
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        has_qr = any(
            b[0] <= (q[0]+q[2])/2 <= b[2] and b[1] <= (q[1]+q[3])/2 <= b[3]
            for q in qr_boxes
        )
        print(f"{i:<4} {int(area):<14} {pct:<10.2f} {int(bw)}x{int(bh):<12} {'EVET' if has_qr else 'yok'}")
