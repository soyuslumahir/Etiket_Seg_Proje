import cv2
import numpy as np
import torch
import random
from pathlib import Path
from ultralytics import YOLO

ETIKET_MODEL_PATH = Path(__file__).parent / "runs/segment/etiket_seg_only/weights/best.pt"
DIGIT_MODEL_PATH  = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"
IMG_DIR  = Path(__file__).parent / "dataset_aug/images"
LBL_DIR  = Path(__file__).parent / "dataset_aug/labels"

print("Modeller yukleniyor...")
etiket_model = YOLO(str(ETIKET_MODEL_PATH))
digit_model  = YOLO(str(DIGIT_MODEL_PATH))
print("Hazir.\n")

imgs = sorted(IMG_DIR.glob("*.jpg"))
ornekler = random.sample(imgs, min(20, len(imgs)))

print(f"{'Dosya':<30} {'Etiket':<10} {'TR':<6} {'Logo':<6} {'QR':<6} {'Rakam'}")
print("-" * 70)

toplam = len(ornekler)
etiket_bulunan = 0

for img_path in sorted(ornekler):
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    img_area = W * H

    res_e = etiket_model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]
    res_d = digit_model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]

    # Etiket sayisi
    n_etiket = 0
    if res_e.masks is not None:
        for conf, box in zip(res_e.boxes.conf, res_e.boxes.xyxy):
            b = box.tolist()
            bw, bh = b[2]-b[0], b[3]-b[1]
            if bw*bh >= img_area*0.003 and 0.3 <= bw/bh <= 3.0:
                n_etiket += 1

    # Diger tespitler
    counts = {1:0, 2:0, 3:0}
    n_rakam = 0
    if res_d.boxes is not None:
        for cls in res_d.boxes.cls:
            cid = int(cls)
            if cid in counts:
                counts[cid] += 1
            elif 4 <= cid <= 13:
                n_rakam += 1

    if n_etiket > 0:
        etiket_bulunan += 1

    torch.cuda.empty_cache()
    print(f"{img_path.name:<30} {n_etiket:<10} {counts[1]:<6} {counts[2]:<6} {counts[3]:<6} {n_rakam}")

print(f"\nEtiket bulma orani: {etiket_bulunan}/{toplam} ({100*etiket_bulunan/toplam:.0f}%)")
