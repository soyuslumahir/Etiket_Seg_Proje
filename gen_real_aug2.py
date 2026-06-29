import cv2, numpy as np, time, shutil
from pathlib import Path
import albumentations as A

SRC_CROP_IMG = Path('real_aug2/crops/images')
SRC_CROP_LBL = Path('real_aug2/crops/labels')
SRC_WIDE_IMG = Path('real_aug/images')
SRC_WIDE_LBL = Path('real_aug/labels')
OUT_IMG = Path('real_aug2/images')
OUT_LBL = Path('real_aug2/labels')
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

N_AUG = 20

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
    coords = ' '.join(f'{max(0.0,min(1.0,x/W)):.6f} {max(0.0,min(1.0,y/H)):.6f}' for x,y in pts)
    return f'{cid} {coords}'

def augment(img_bgr, ann, seed):
    rng = np.random.RandomState(seed)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # Dikey flip
    if rng.random() < 0.5:
        img = img[::-1].copy()
        ann = [(cid, [(x, H-1-y) for x,y in pts]) for cid,pts in ann]

    # +-5 derece rotasyon
    angle = rng.uniform(-5, 5)
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W,H), borderMode=cv2.BORDER_REFLECT_101)
        new_ann = []
        for cid, pts in ann:
            new_pts = []
            for x, y in pts:
                nx = M[0,0]*x + M[0,1]*y + M[0,2]
                ny = M[1,0]*x + M[1,1]*y + M[1,2]
                new_pts.append((max(0.0, min(W-1, nx)), max(0.0, min(H-1, ny))))
            new_ann.append((cid, new_pts))
        ann = new_ann

    # Photometric
    img = A.Compose([
        A.RandomBrightnessContrast(0.12, 0.12, p=0.7),
        A.HueSaturationValue(5, 12, 10, p=0.6),
        A.RandomGamma((90, 115), p=0.4),
        A.GaussNoise(var_limit=(2, 8), p=0.5),
        A.MotionBlur(blur_limit=(3, 5), p=0.1),
        A.ImageCompression(quality_lower=75, quality_upper=98, p=0.4),
    ])(image=img)['image']

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), ann

done = 0
t0 = time.time()

# --- 3 orijinal wide foto (kendileri) ---
wide_imgs = sorted(SRC_WIDE_IMG.glob('usb_frame*.jpg'))
for img_path in wide_imgs:
    lbl_path = SRC_WIDE_LBL / (img_path.stem + '.txt')
    shutil.copy(img_path, OUT_IMG / img_path.name)
    shutil.copy(lbl_path, OUT_LBL / (img_path.stem + '.txt'))
print(f'{len(wide_imgs)} orijinal wide foto kopyalandi')

# --- Toplam kaynak sayisi: 36 crop + 3 wide = 39 ---
sources = []
for p in sorted(SRC_CROP_IMG.glob('*.jpg')):
    sources.append(('crop', p, SRC_CROP_LBL / (p.stem + '.txt')))
for p in wide_imgs:
    sources.append(('wide', p, SRC_WIDE_LBL / (p.stem + '.txt')))

total = len(sources) * N_AUG
print(f'{len(sources)} kaynak x {N_AUG} aug = {total} goruntu uretilecek')

for src_type, img_path, lbl_path in sources:
    img_bgr = cv2.imread(str(img_path))
    H, W = img_bgr.shape[:2]
    ann = yolo_to_px(lbl_path.read_text().strip().split('\n'), W, H)

    for ai in range(N_AUG):
        seed = abs(hash(img_path.stem)) % (2**31) + ai * 1009
        aug_img, aug_ann = augment(img_bgr, ann, seed)
        aH, aW = aug_img.shape[:2]

        out_stem = f'{img_path.stem}_a{ai:02d}'
        cv2.imwrite(str(OUT_IMG / f'{out_stem}.jpg'), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [px_to_yolo(cid, pts, aW, aH) for cid, pts in aug_ann if len(pts) >= 3]
        (OUT_LBL / f'{out_stem}.txt').write_text('\n'.join(lines))

        done += 1
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 1
        remain = (total - done) / rate
        print(f'{done}/{total}  kalan: {remain/60:.1f} dk', end='\r')

total_img = len(list(OUT_IMG.glob('*.jpg')))
total_lbl = len(list(OUT_LBL.glob('*.txt')))
print(f'\nBitti! {total_img} goruntu, {total_lbl} label -> real_aug2/')
