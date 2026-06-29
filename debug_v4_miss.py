import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"
model = YOLO(str(MODEL_PATH))

# Kaçan görüntüler
imgs = [
    "tip2_0254_a0.jpg",
    "tip2_0260_a1.jpg",
    "tip2_0336_a0.jpg",
    "tip1_0163_a0.jpg",
]

for name in imgs:
    img_path = Path(__file__).parent / "dataset_aug/images" / name
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    img_area = W * H

    res = model(img_bgr, conf=0.01, imgsz=1920, verbose=False)[0]
    print(f"\n{name}  ({W}x{H})")

    if res.boxes is None or len(res.boxes) == 0:
        print("  Hic tespit yok")
        continue

    etiket_detections = [(float(conf), box.tolist())
                         for cls, conf, box in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)
                         if int(cls) == 0]

    print(f"  {'conf':<8} {'alan%':<10} {'aspect':<10} {'filtre'}")
    for conf, b in sorted(etiket_detections, reverse=True):
        bw, bh = b[2]-b[0], b[3]-b[1]
        alan_pct = bw*bh / img_area * 100
        aspect = bw/bh if bh > 0 else 0
        reasons = []
        if conf < 0.25: reasons.append(f"conf={conf:.3f}")
        if bw*bh < img_area*0.003: reasons.append(f"alan={alan_pct:.1f}%")
        if not (0.3 <= aspect <= 3.0): reasons.append(f"aspect={aspect:.2f}")
        durum = "GECTI" if not reasons else "FILTRE: " + ", ".join(reasons)
        print(f"  {conf:<8.3f} {alan_pct:<10.1f} {aspect:<10.2f} {durum}")

    torch.cuda.empty_cache()
