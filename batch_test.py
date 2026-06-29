import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

ETIKET_MODEL_PATH = Path(__file__).parent / "runs/segment/etiket_seg_only/weights/best.pt"
DIGIT_MODEL_PATH  = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"
TEST_DIR  = Path(__file__).parent / "test_gercek_aug"
OUT_DIR   = Path(__file__).parent / "test_gercek_aug_out"
OUT_DIR.mkdir(exist_ok=True)

CC = {
    0:(200,200,200), 1:(0,200,0), 2:(0,100,255), 3:(255,165,0),
    4:(255,50,50), 5:(255,150,50), 6:(255,230,50), 7:(50,255,50),
    8:(50,255,200), 9:(50,200,255), 10:(50,50,255), 11:(200,50,255),
    12:(255,50,200), 13:(180,180,100)
}
NAMES = {
    0:'etiket', 1:'TR', 2:'logo', 3:'QR',
    4:'0', 5:'1', 6:'2', 7:'3', 8:'4',
    9:'5', 10:'6', 11:'7', 12:'8', 13:'9'
}

print("Modeller yukleniyor...")
etiket_model = YOLO(str(ETIKET_MODEL_PATH))
digit_model  = YOLO(str(DIGIT_MODEL_PATH))
print("Hazir.\n")


def digits_per_tag(etiket_items, other_detections, H, W):
    if not etiket_items:
        return []

    etiket_boxes   = [item[0] for item in etiket_items]
    etiket_masks   = [item[1] for item in etiket_items]
    etiket_centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in etiket_boxes]

    rakamlar   = [((b[0]+b[2])/2, (b[1]+b[3])/2, cid-4)
                  for cid, conf, b in other_detections if 4 <= cid <= 13]
    tr_pts_all = [((b[0]+b[2])/2, (b[1]+b[3])/2)
                  for cid, conf, b in other_detections if cid == 1]

    def nearest_etiket(px, py):
        px_i = int(min(max(px, 0), W-1))
        py_i = int(min(max(py, 0), H-1))
        icinde = [i for i, m in enumerate(etiket_masks) if m[py_i, px_i] > 0]
        if len(icinde) == 1:
            return icinde[0]
        candidates = icinde if icinde else range(len(etiket_centers))
        return min(candidates, key=lambda i: (px-etiket_centers[i][0])**2 + (py-etiket_centers[i][1])**2)

    gruplar    = {i: [] for i in range(len(etiket_items))}
    tr_gruplar = {i: [] for i in range(len(etiket_items))}

    for rx, ry, d in rakamlar:
        gruplar[nearest_etiket(rx, ry)].append((rx, ry, d))
    for hx, hy in tr_pts_all:
        tr_gruplar[nearest_etiket(hx, hy)].append((hx, hy))

    sonuclar = []
    for i in sorted(range(len(etiket_items)), key=lambda j: etiket_boxes[j][0]):
        icindekiler = gruplar[i]
        if len(icindekiler) < 2:
            sonuclar.append("".join(str(d) for _, _, d in icindekiler))
            continue

        pts  = np.array([[rx, ry] for rx, ry, _ in icindekiler], dtype=np.float32)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        axis = vt[0]

        projections   = [float(np.dot(np.array([rx, ry]) - mean, axis)) for rx, ry, _ in icindekiler]
        sorted_digits = sorted(zip(projections, [d for _, _, d in icindekiler]))

        if tr_gruplar[i]:
            tr_x = sum(p[0] for p in tr_gruplar[i]) / len(tr_gruplar[i])
            tr_y = sum(p[1] for p in tr_gruplar[i]) / len(tr_gruplar[i])
            if float(np.dot(np.array([tr_x, tr_y]) - mean, axis)) > 0:
                sorted_digits = sorted_digits[::-1]

        sonuclar.append("".join(str(d) for _, d in sorted_digits))

    return sonuclar


def test_image(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, "OKUNAMADI"
    H, W = img_bgr.shape[:2]

    min_etiket_area = W * H * 0.003
    res_etiket = etiket_model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]

    etiket_items = []
    out     = img_bgr.copy()
    overlay = out.copy()

    if res_etiket.masks is not None:
        for mask, conf, box in zip(res_etiket.masks.data, res_etiket.boxes.conf,
                                   res_etiket.boxes.xyxy):
            b      = box.tolist()
            bw, bh = b[2]-b[0], b[3]-b[1]
            if bw*bh < min_etiket_area or not (0.3 <= bw/bh <= 3.0):
                continue
            m = cv2.resize(mask.cpu().numpy(), (W, H), interpolation=cv2.INTER_NEAREST)
            binary = (m > 0.5).astype(np.uint8)
            k      = max(5, int(min(bw, bh) / 8))
            filled = cv2.dilate(binary, np.ones((k, k), np.uint8))
            etiket_items.append((b, filled))
            overlay[binary > 0] = (60, 60, 60)

    res_digit = digit_model(img_bgr, conf=0.25, imgsz=2048, verbose=False)[0]
    other_detections = []

    if res_digit.masks is not None:
        for mask, cls, conf, box in zip(res_digit.masks.data, res_digit.boxes.cls,
                                        res_digit.boxes.conf, res_digit.boxes.xyxy):
            cid = int(cls)
            if cid == 0:
                continue
            b     = box.tolist()
            color = CC.get(cid, (200, 200, 200))
            m     = cv2.resize(mask.cpu().numpy(), (W, H), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0.5] = color
            other_detections.append((cid, float(conf), b))

    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    numaralar = digits_per_tag(etiket_items, other_detections, H, W)

    # Etikete numara yaz
    for i, (b, _) in enumerate(etiket_items):
        cx = int((b[0]+b[2])/2)
        cy = int((b[1]+b[3])/2)
        numara = numaralar[i] if i < len(numaralar) else "?"
        cv2.putText(out, numara, (cx-60, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60,60,60), 2)

    torch.cuda.empty_cache()
    return out, numaralar


imgs = sorted(TEST_DIR.glob("*.jpg"))
print(f"Toplam: {len(imgs)} goruntu\n")
print(f"{'Dosya':<45} Sonuc")
print("-" * 65)

for img_path in imgs:
    out_img, numaralar = test_image(img_path)
    if out_img is not None:
        out_path = OUT_DIR / img_path.name
        cv2.imwrite(str(out_path), out_img)
    numara_str = "  |  ".join(numaralar) if numaralar else "(etiket bulunamadi)"
    print(f"{img_path.name:<45} {numara_str}")

print(f"\nAnnotated goruntular: {OUT_DIR}")
