import cv2, numpy as np, random, json
from pathlib import Path
import albumentations as A

REAL_IMG = Path('real_aug/images')
REAL_LBL = Path('real_aug/labels')
CROPS_DIR = Path('real_aug/crops')
OUT_IMG   = Path('real_aug/aug_images')
OUT_LBL   = Path('real_aug/aug_labels')
OUT_IMG.mkdir(exist_ok=True)
OUT_LBL.mkdir(exist_ok=True)

N_AUG = 20
PAD   = 10   # crop kesme padding (gen_2000 ile ayni)


# ── Yardimci: YOLO polygon satirlarini piksel noktalarına cevir ──
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
    coords = ' '.join(f'{x/W:.6f} {y/H:.6f}' for x, y in pts)
    return f'{cid} {coords}'

def clip_pts(pts, W, H):
    return [(max(0.0, min(W-1, x)), max(0.0, min(H-1, y))) for x, y in pts]


# ── Augmentation: image only (renk/noise) ──
def color_aug(img_rgb, rng):
    transforms = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=10, p=0.6),
        A.RandomGamma(gamma_limit=(90, 115), p=0.4),
        A.GaussNoise(var_limit=(2.0, 8.0), p=0.5),
        A.MotionBlur(blur_limit=(3, 5), p=0.1),
        A.ImageCompression(quality_lower=75, quality_upper=98, p=0.4),
    ])
    return transforms(image=img_rgb)['image']


# ── Augmentation: image + label dönüşümü ──
def augment_labeled(img_bgr, ann, rng):
    """
    ann: list of (cid, [(px,py),...])
    Returns: (aug_bgr, aug_ann)
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 1. Vertical flip (%50)
    do_flip = rng.random() < 0.5
    if do_flip:
        img = img[::-1, :, :].copy()
        ann = [(cid, [(x, H-1-y) for x, y in pts]) for cid, pts in ann]

    # 2. Küçük rotation ±5°
    angle = rng.uniform(-5, 5)
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        new_ann = []
        for cid, pts in ann:
            new_pts = []
            for x, y in pts:
                xc, yc = x - W/2, y - H/2
                xr = cos_a*xc - sin_a*yc + W/2
                yr = sin_a*xc + cos_a*yc + H/2
                new_pts.append((xr, yr))
            new_pts = clip_pts(new_pts, W, H)
            new_ann.append((cid, new_pts))
        ann = new_ann

    # 3. Renk/noise (sadece görsel)
    img = color_aug(img, rng)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), ann


# ── Crop labellarını hesapla ──
def crop_labels(parent_ann, img_W, img_H, x1, y1, x2, y2):
    """Parent fotoğraf labellarından crop'a düsen kısmı çıkar."""
    cw, ch = x2 - x1, y2 - y1
    result = []
    for cid, pts in parent_ann:
        # Merkezi crop içinde mi?
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        # Koordinatları cropa göre dönüştür
        new_pts = [(x - x1, y - y1) for x, y in pts]
        new_pts = clip_pts(new_pts, cw, ch)
        result.append((cid, new_pts))
    return result, cw, ch


# ══════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ══════════════════════════════════════════════════════════════════
done = 0
wide_imgs = sorted(REAL_IMG.glob('*.jpg'))

# ── A) 3 geniş fotoğraf ──
for img_path in wide_imgs:
    lbl_path = REAL_LBL / (img_path.stem + '.txt')
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    lines = lbl_path.read_text().strip().split('\n')
    ann = yolo_to_px(lines, W, H)

    for ai in range(N_AUG):
        rng = np.random.RandomState(abs(hash(img_path.name)) % (2**31) + ai * 1009)
        stem = f'wide_{img_path.stem}_a{ai:02d}'
        aug_bgr, aug_ann = augment_labeled(img_bgr, ann, rng)
        aH, aW = aug_bgr.shape[:2]

        cv2.imwrite(str(OUT_IMG / f'{stem}.jpg'), aug_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        yolo_lines = [px_to_yolo(cid, pts, aW, aH) for cid, pts in aug_ann if len(pts) >= 3]
        (OUT_LBL / f'{stem}.txt').write_text('\n'.join(yolo_lines))
        done += 1

print(f'Genis fotolar: {done} uretildi')

# ── B) 36 crop ──
crop_files = sorted(CROPS_DIR.glob('*.jpg'))
crop_done = 0

for crop_path in crop_files:
    # Parent tespiti: stem = {parent_stem}_etiket_{idx}
    stem_parts = crop_path.stem.rsplit('_etiket_', 1)
    parent_stem = stem_parts[0]
    etiket_idx  = int(stem_parts[1])

    parent_img_path = REAL_IMG / (parent_stem + '.jpg')
    parent_lbl_path = REAL_LBL / (parent_stem + '.txt')
    if not parent_img_path.exists() or not parent_lbl_path.exists():
        continue

    parent_img = cv2.imread(str(parent_img_path))
    pH, pW = parent_img.shape[:2]
    parent_lines = parent_lbl_path.read_text().strip().split('\n')
    parent_ann = yolo_to_px(parent_lines, pW, pH)

    # Crop bbox: etiket_idx'inci class-0 polygonundan
    etiket_polys = [(cid, pts) for cid, pts in parent_ann if cid == 0]
    if etiket_idx >= len(etiket_polys):
        continue
    _, bbox_pts = etiket_polys[etiket_idx]
    xs = [p[0] for p in bbox_pts]; ys = [p[1] for p in bbox_pts]
    x1 = max(0, int(min(xs)) - PAD); y1 = max(0, int(min(ys)) - PAD)
    x2 = min(pW, int(max(xs)) + PAD); y2 = min(pH, int(max(ys)) + PAD)

    crop_bgr = cv2.imread(str(crop_path))
    crop_ann, cw, ch = crop_labels(parent_ann, pW, pH, x1, y1, x2, y2)

    for ai in range(N_AUG):
        rng = np.random.RandomState(abs(hash(crop_path.name)) % (2**31) + ai * 997)
        out_stem = f'crop_{crop_path.stem}_a{ai:02d}'
        aug_bgr, aug_ann = augment_labeled(crop_bgr, crop_ann, rng)
        aH, aW = aug_bgr.shape[:2]

        cv2.imwrite(str(OUT_IMG / f'{out_stem}.jpg'), aug_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        yolo_lines = [px_to_yolo(cid, pts, aW, aH) for cid, pts in aug_ann if len(pts) >= 3]
        (OUT_LBL / f'{out_stem}.txt').write_text('\n'.join(yolo_lines))
        crop_done += 1

    done += N_AUG

print(f'Croplar: {crop_done} uretildi')
print(f'TOPLAM: {done} goruntu')
