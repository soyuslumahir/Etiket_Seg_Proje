import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = 'runs/segment/runs/seg/v5-8/weights/best.pt'
RAW_DIR    = Path('raw_images_4000x3000')
OUT_DIR    = Path('test_raw_output')
OUT_DIR.mkdir(exist_ok=True)

CC = {
    0:(200,200,200), 1:(0,200,0), 2:(0,100,255), 3:(255,165,0),
    4:(255,50,50), 5:(255,150,50), 6:(255,230,50), 7:(50,255,50),
    8:(50,255,200), 9:(50,200,255), 10:(50,50,255), 11:(200,50,255),
    12:(255,50,200), 13:(180,180,100)
}
NAMES = {
    0:'etiket', 1:'TR', 2:'logo', 3:'QR',
    4:'rakam_0', 5:'rakam_1', 6:'rakam_2', 7:'rakam_3',
    8:'rakam_4', 9:'rakam_5', 10:'rakam_6', 11:'rakam_7',
    12:'rakam_8', 13:'rakam_9'
}

print("Model yukleniyor...")
model = YOLO(MODEL_PATH)
print("Hazir.\n")


def clean_etiket_poly(poly, H, W, scale=4):
    sh, sw = H // scale, W // scale
    pts = (poly / scale).astype(np.int32)
    mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    k = 5
    kernel = np.ones((k, k), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return poly.astype(np.float32)
    largest = max(cnts, key=cv2.contourArea)
    return (largest.reshape(-1, 2) * scale).astype(np.float32)


def digits_per_tag(detections, polys, H, W):
    etiket_polys = []
    etiket_boxes = []
    for (cid, conf, box), poly in zip(detections, polys):
        if cid != 0:
            continue
        cleaned = clean_etiket_poly(poly, H, W)
        etiket_polys.append(cleaned)
        etiket_boxes.append(box)

    if not etiket_polys:
        return []

    etiket_centers = [(float(poly[:, 0].mean()), float(poly[:, 1].mean())) for poly in etiket_polys]

    rakamlar = [((b[0]+b[2])/2, (b[1]+b[3])/2, cid-4, conf, b)
                for cid, conf, b in detections if 4 <= cid <= 13]
    header_pts_all = [((b[0]+b[2])/2, (b[1]+b[3])/2)
                      for cid, conf, b in detections if cid in (1, 2, 3)]

    MARGIN = 15

    def nearest_etiket(px, py):
        inside = [i for i, poly in enumerate(etiket_polys)
                  if cv2.pointPolygonTest(poly, (float(px), float(py)), True) >= MARGIN]
        if len(inside) == 1:
            return inside[0]
        if inside:
            return min(inside, key=lambda i: (px-etiket_centers[i][0])**2 + (py-etiket_centers[i][1])**2)
        inside_loose = [i for i, poly in enumerate(etiket_polys)
                        if cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0]
        if inside_loose:
            return min(inside_loose, key=lambda i: (px-etiket_centers[i][0])**2 + (py-etiket_centers[i][1])**2)
        return min(range(len(etiket_centers)),
                   key=lambda i: (px-etiket_centers[i][0])**2 + (py-etiket_centers[i][1])**2)

    gruplar        = {i: [] for i in range(len(etiket_polys))}
    header_gruplar = {i: [] for i in range(len(etiket_polys))}

    for rx, ry, d, conf, b in rakamlar:
        gruplar[nearest_etiket(rx, ry)].append((rx, ry, d, conf, b))

    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    IOU_THRESH = 0.3
    for i in gruplar:
        items = sorted(gruplar[i], key=lambda x: -x[3])
        kept = []
        for item in items:
            if not any(box_iou(item[4], k[4]) > IOU_THRESH for k in kept):
                kept.append(item)
        gruplar[i] = kept

    for hx, hy in header_pts_all:
        inside = [i for i, poly in enumerate(etiket_polys)
                  if cv2.pointPolygonTest(poly, (float(hx), float(hy)), True) >= MARGIN]
        if not inside:
            inside = [i for i, poly in enumerate(etiket_polys)
                      if cv2.pointPolygonTest(poly, (float(hx), float(hy)), False) >= 0]
        if inside:
            idx = min(inside, key=lambda i: (hx-etiket_centers[i][0])**2 + (hy-etiket_centers[i][1])**2)
            header_gruplar[idx].append((hx, hy))

    sonuclar = []
    for i in sorted(range(len(etiket_polys)), key=lambda j: etiket_centers[j][0]):
        icindekiler = gruplar[i]
        if len(icindekiler) < 2:
            sonuclar.append("".join(str(d) for _, _, d, _, _ in icindekiler))
            continue

        pts  = np.array([[rx, ry] for rx, ry, _, _, _ in icindekiler], dtype=np.float32)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        axis = vt[0]
        if axis[0] < 0:
            axis = -axis

        projections   = [float(np.dot(np.array([rx, ry]) - mean, axis)) for rx, ry, _, _, _ in icindekiler]
        sorted_digits = sorted(zip(projections, [d for _, _, d, _, _ in icindekiler]))

        digit_spread = max(projections) - min(projections) if len(projections) > 1 else 1
        header_used = False
        if header_gruplar[i]:
            hx = sum(p[0] for p in header_gruplar[i]) / len(header_gruplar[i])
            hy = sum(p[1] for p in header_gruplar[i]) / len(header_gruplar[i])
            hproj = float(np.dot(np.array([hx, hy]) - mean, axis))
            if hproj >= digit_spread * 0.04:
                sorted_digits = sorted_digits[::-1]
                header_used = True

        numara = "".join(str(d) for _, d in sorted_digits)

        if not header_used:
            trailing = len(numara) - len(numara.rstrip('0'))
            leading  = len(numara) - len(numara.lstrip('0'))
            if trailing > leading:
                numara = numara[::-1]

        sonuclar.append(numara)
    return sonuclar


def run_on_image(img_path):
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    print(f"  Goruntu boyutu: {W}x{H}")

    imgsz = 1280
    result     = model(img_bgr, conf=0.25, imgsz=imgsz, verbose=False)[0]
    result_hdr = model(img_bgr, conf=0.10, imgsz=imgsz, verbose=False)[0]

    out     = img_bgr.copy()
    overlay = out.copy()
    detections     = []
    filtered_masks = []
    min_etiket_area = W * H * 0.003

    if result.masks is not None:
        polys = result.masks.xy
        for poly, cls, conf, box in zip(polys, result.boxes.cls,
                                        result.boxes.conf, result.boxes.xyxy):
            cid = int(cls)
            b   = box.tolist()
            if cid == 0:
                bw = b[2]-b[0]; bh = b[3]-b[1]
                area = bw*bh; aspect = bw/bh if bh > 0 else 0
                if area < min_etiket_area or float(conf) < 0.4 or not (0.3 <= aspect <= 3.0):
                    continue
            color     = CC.get(cid, (200, 200, 200))
            draw_poly = clean_etiket_poly(poly, H, W) if cid == 0 else poly
            pts       = draw_poly.astype(np.int32)
            tmp = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(tmp, [pts], 1)
            cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                cv2.fillPoly(overlay, [cnt.reshape(-1, 2)], color)
            detections.append((cid, float(conf), b))
            filtered_masks.append(draw_poly)

        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
        for poly, (cid, conf, b) in zip(filtered_masks, detections):
            color = CC.get(cid, (200, 200, 200))
            pts   = poly.astype(np.int32)
            cv2.polylines(out, [pts], True, color, 2)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(out, f"{NAMES.get(cid,'?')} {conf:.2f}",
                        (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    hdr_detections = []
    if result_hdr.boxes is not None:
        for cls, conf, box in zip(result_hdr.boxes.cls, result_hdr.boxes.conf, result_hdr.boxes.xyxy):
            cid = int(cls)
            if cid in (1, 2, 3):
                hdr_detections.append((cid, float(conf), box.tolist()))

    numaralar = digits_per_tag(detections + hdr_detections, filtered_masks, H, W) if filtered_masks else []

    n_etiket = sum(1 for cid, _, _ in detections if cid == 0)
    counts = {}
    for cid, conf, _ in detections:
        counts[NAMES.get(cid, str(cid))] = counts.get(NAMES.get(cid, str(cid)), 0) + 1

    print(f"  Tespit edilen etiket: {n_etiket}")
    for i, numara in enumerate(numaralar, 1):
        print(f"    Etiket {i}: {numara if numara else '(rakam yok)'}")
    print(f"  Sinif sayilari: { {k:v for k,v in sorted(counts.items())} }")

    # Sonuc gorselini kaydet (1/4 olcekle — 4000x3000 -> 1000x750)
    scale_out = 0.25
    out_small = cv2.resize(out, (int(W*scale_out), int(H*scale_out)))
    out_path  = OUT_DIR / (img_path.stem + '_result.jpg')
    cv2.imwrite(str(out_path), out_small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Kaydedildi: {out_path}")

    return numaralar, n_etiket


print("=" * 60)
img_paths = sorted(RAW_DIR.glob('*.jpg'))
all_results = {}

for img_path in img_paths:
    print(f"\n[{img_path.name}]")
    numaralar, n_etiket = run_on_image(img_path)
    all_results[img_path.name] = numaralar

print("\n" + "=" * 60)
print("OZET — Okunan Numaralar:")
print("=" * 60)
for fname, numaralar in all_results.items():
    print(f"{fname}:")
    for i, n in enumerate(numaralar, 1):
        print(f"  Etiket {i}: {n}")
print("=" * 60)
print(f"\nCikti gorseller: {OUT_DIR}/")
