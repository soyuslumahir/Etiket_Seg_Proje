import cv2, numpy as np
from pathlib import Path

SRC_IMG = Path('real_aug/images')
SRC_LBL = Path('real_aug/labels')
OUT_IMG  = Path('real_aug2/crops/images')
OUT_LBL  = Path('real_aug2/crops/labels')
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

PAD = 40  # piksel padding

def yolo_to_px(lines, W, H):
    result = []
    for line in lines:
        if not line.strip(): continue
        parts = line.split()
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = [(coords[i]*W, coords[i+1]*H) for i in range(0, len(coords), 2)]
        result.append((cid, pts))
    return result

def px_to_yolo(cid, pts, W, H):
    coords = ' '.join(f'{max(0.0,min(1.0,x/W)):.6f} {max(0.0,min(1.0,y/H)):.6f}' for x, y in pts)
    return f'{cid} {coords}'

originals = sorted(SRC_IMG.glob('usb_frame*.jpg'))
total_crops = 0

for img_path in originals:
    lbl_path = SRC_LBL / (img_path.stem + '.txt')
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    lines = [l for l in lbl_path.read_text().strip().split('\n') if l]
    ann = yolo_to_px(lines, W, H)

    # Sadece class-0 (etiket siniri) polygonlari bul
    etiket_polys = [(i, pts) for i, (cid, pts) in enumerate(ann) if cid == 0]

    for idx, (ann_idx, etk_pts) in enumerate(etiket_polys):
        xs = [p[0] for p in etk_pts]
        ys = [p[1] for p in etk_pts]
        x1 = max(0, int(min(xs)) - PAD)
        y1 = max(0, int(min(ys)) - PAD)
        x2 = min(W, int(max(xs)) + PAD)
        y2 = min(H, int(max(ys)) + PAD)

        crop = img[y1:y2, x1:x2]
        cw, ch = x2 - x1, y2 - y1

        # Crop icindeki tum annotationlari hesapla
        crop_lines = []
        for cid, pts in ann:
            # Merkezi crop icinde mi?
            cx = np.mean([p[0] for p in pts])
            cy = np.mean([p[1] for p in pts])
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            new_pts = [(x - x1, y - y1) for x, y in pts]
            crop_lines.append(px_to_yolo(cid, new_pts, cw, ch))

        stem = f'{img_path.stem}_etiket_{idx:02d}'
        cv2.imwrite(str(OUT_IMG / f'{stem}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT_LBL / f'{stem}.txt').write_text('\n'.join(crop_lines))
        total_crops += 1
        print(f'  {stem}: {cw}x{ch}  {len(crop_lines)} annotation')

print(f'\nToplam: {total_crops} crop uretildi -> real_aug2/crops/')
