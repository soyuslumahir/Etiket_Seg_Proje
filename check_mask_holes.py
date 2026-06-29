import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"
model = YOLO(str(MODEL_PATH))

imgs = [
    Path(__file__).parent / "test_gercek_aug/usb_frame_20260622_103722_033.jpg",
    Path(__file__).parent / "test_gercek_aug/usb_frame_20260622_103815_642.jpg",
]

for img_path in imgs:
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    img_area = W * H

    res = model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]
    print(f"\n{img_path.name}")
    print(f"  {'#':<4} {'Delik%':<12} {'Parcalar':<12} {'Durum'}")
    print("  " + "-" * 45)

    idx = 0
    if res.masks is not None:
        for mask, cls, conf, box in zip(res.masks.data, res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
            if int(cls) != 0:
                continue
            b = box.tolist()
            bw, bh = b[2]-b[0], b[3]-b[1]
            if bw*bh < img_area*0.003 or not (0.3 <= bw/bh <= 3.0):
                continue
            idx += 1

            m = cv2.resize(mask.cpu().numpy(), (W, H), interpolation=cv2.INTER_NEAREST)
            binary = (m > 0.5).astype(np.uint8)

            # Dis konturu bul ve icini tamamen doldur
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(binary)
            cv2.drawContours(filled, cnts, -1, 1, -1)  # ic dolu

            mask_px   = int(binary.sum())
            filled_px = int(filled.sum())
            delik_px  = filled_px - mask_px
            delik_pct = 100 * delik_px / filled_px if filled_px > 0 else 0

            n_comp, _ = cv2.connectedComponents(binary)
            n_comp -= 1

            durum = "OK" if delik_pct < 5 else "DELIKLI"
            print(f"  {idx:<4} {delik_pct:<12.1f} {n_comp:<12} {durum}")

    torch.cuda.empty_cache()
