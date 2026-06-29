import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

ETIKET_MODEL_PATH = Path(__file__).parent / "runs/segment/etiket_seg_only/weights/best.pt"
model = YOLO(str(ETIKET_MODEL_PATH))

imgs = [
    Path(__file__).parent / "dataset_aug/images/tip1_0188_a0.jpg",
    Path(__file__).parent / "dataset_aug/images/tip2_0255_a1.jpg",
]

for img_path in imgs:
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    img_area = W * H
    print(f"\n{img_path.name}  ({W}x{H})")

    res = model(img_bgr, conf=0.01, imgsz=1920, verbose=False)[0]  # conf=0.01 ile hepsini goster

    if res.boxes is None or len(res.boxes) == 0:
        print("  Hic tespit yok (conf=0.01 ile bile)")
        continue

    print(f"  {'conf':<8} {'alan%':<10} {'BxH':<18} {'aspect':<10} {'filtre durumu'}")
    for conf, box in zip(res.boxes.conf, res.boxes.xyxy):
        b  = box.tolist()
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        area = bw * bh
        aspect = bw / bh if bh > 0 else 0
        pct = area / img_area * 100

        reasons = []
        if float(conf) < 0.25:   reasons.append(f"conf<0.25({float(conf):.3f})")
        if area < img_area*0.003: reasons.append(f"alan<0.3%({pct:.2f}%)")
        if not (0.3 <= aspect <= 3.0): reasons.append(f"aspect={aspect:.2f}")
        durum = "GECTI" if not reasons else "FILTRENDI: " + ", ".join(reasons)

        print(f"  {float(conf):<8.3f} {pct:<10.2f} {int(bw)}x{int(bh):<12} {aspect:<10.2f} {durum}")

    torch.cuda.empty_cache()
