import cv2
import torch
import random
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"
IMG_DIR    = Path(__file__).parent / "dataset_aug/images"
LBL_DIR    = Path(__file__).parent / "dataset_aug/labels"

model = YOLO(str(MODEL_PATH))

imgs = sorted(IMG_DIR.glob("*.jpg"))
ornekler = random.sample(imgs, min(20, len(imgs)))

print(f"{'Dosya':<30} {'GT_etiket':<12} {'Bulunan':<10} {'GT_rakam':<12} {'Bulunan':<10} {'Durum'}")
print("-" * 85)

toplam = len(ornekler)
dogru = 0

for img_path in sorted(ornekler):
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    img_area = W * H

    # Ground truth label oku
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    gt_etiket = 0
    gt_rakam = 0
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split('\n'):
            if not line: continue
            cid = int(line.split()[0])
            if cid == 0: gt_etiket += 1
            elif 4 <= cid <= 13: gt_rakam += 1

    # Model tahmini
    res = model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]
    pred_etiket = 0
    pred_rakam = 0
    if res.boxes is not None:
        for cls, conf, box in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
            cid = int(cls)
            if cid == 0:
                b = box.tolist()
                bw, bh = b[2]-b[0], b[3]-b[1]
                if bw*bh >= img_area*0.003 and 0.3 <= bw/bh <= 3.0:
                    pred_etiket += 1
            elif 4 <= cid <= 13:
                pred_rakam += 1

    ok = pred_etiket == gt_etiket
    if ok: dogru += 1
    durum = "OK" if ok else f"MISS(gt={gt_etiket},pred={pred_etiket})"
    print(f"{img_path.name:<30} {gt_etiket:<12} {pred_etiket:<10} {gt_rakam:<12} {pred_rakam:<10} {durum}")
    torch.cuda.empty_cache()

print(f"\nEtiket dogru tespit: {dogru}/{toplam} ({100*dogru/toplam:.0f}%)")
