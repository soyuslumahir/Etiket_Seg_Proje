import cv2, numpy as np, random, time
from pathlib import Path
from PIL import Image
import albumentations as A

CC = {
    0:(200,200,200), 1:(0,200,0),   2:(0,100,255),  3:(255,165,0),
    4:(255,50,50),   5:(255,150,50), 6:(255,230,50), 7:(50,255,50),
    8:(50,255,200),  9:(50,200,255), 10:(50,50,255), 11:(200,50,255),
    12:(255,50,200), 13:(180,180,100)
}

def mask_to_yolo(msk_rgb, W, H):
    lines = []
    for cid, color in CC.items():
        if cid == 0:
            binary = (msk_rgb.max(axis=2) > 10).astype(np.uint8) * 255
            min_area = 2000
        else:
            c = np.array(color, dtype=np.int32)
            diff = np.abs(msk_rgb.astype(np.int32) - c).max(axis=2)
            binary = (diff < 5).astype(np.uint8) * 255
            min_area = 100
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            eps = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            coords = ' '.join(f'{x/W:.6f} {y/H:.6f}' for x, y in approx)
            lines.append(f'{cid} {coords}')
    return lines

IMG_DIR = Path('scenes/images')
MSK_DIR = Path('scenes/masks')
LBL_DIR = Path('scenes/labels')

BG_RGB = (209, 196, 38)
N_AUG  = 2   # her sahne icin augmentation sayisi

def augment_scene(img_bgr, msk_rgb, seed):
    rng = np.random.RandomState(seed)
    random.seed(seed)

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    msk = msk_rgb.copy()

    # Sahne rotasyonu (discrete 90° adimlar)
    angle = int(rng.choice([0, 90, 180, 270]))
    if angle != 0:
        H0, W0 = img.shape[:2]
        Mc = cv2.getRotationMatrix2D((W0//2, H0//2), angle, 1.0)
        img = cv2.warpAffine(img, Mc, (W0, H0), borderMode=cv2.BORDER_CONSTANT, borderValue=BG_RGB)
        msk = cv2.warpAffine(msk, Mc, (W0, H0), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Lighting & color
    stage3 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.7),
        A.HueSaturationValue(hue_shift_limit=7, sat_shift_limit=18, val_shift_limit=12, p=0.7),
        A.RandomGamma(gamma_limit=(85, 120), p=0.5),
    ])
    img = stage3(image=img)['image']

    if rng.random() < 0.65:
        yellow = np.full_like(img, (220, 200, 50), dtype=np.uint8)
        img = cv2.addWeighted(img, 1 - rng.uniform(0.03, 0.09), yellow, rng.uniform(0.03, 0.09), 0)

    # Noise & blur
    stage4 = A.Compose([
        A.GaussNoise(var_limit=(3.0, 12.0), p=0.6),
        A.MotionBlur(blur_limit=(3, 5), p=0.15),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),
    ])
    img = stage4(image=img)['image']

    # Specular highlights
    tag_any = msk.max(axis=2) > 10
    ys, xs = np.where(tag_any)
    if len(xs) > 0:
        for _ in range(rng.randint(2, 6)):
            idx = rng.randint(0, len(xs))
            spec = np.zeros_like(img)
            rx, ry = rng.randint(15, 50), rng.randint(15, 50)
            cv2.ellipse(spec, (int(xs[idx]), int(ys[idx])), (rx, ry),
                        int(rng.uniform(0, 360)), 0, 360, (255, 255, 255), -1)
            spec = cv2.GaussianBlur(spec, (41, 41), 0)
            img = cv2.addWeighted(img, 1.0, spec, rng.uniform(0.05, 0.16), 0)

    # Shadow
    if rng.random() < 0.65:
        H2, W2 = img.shape[:2]
        tag_bin = (msk.max(axis=2) > 10).astype(np.uint8)
        ox_s, oy_s = rng.randint(5, 18), rng.randint(5, 18)
        sh = np.zeros((H2 + oy_s, W2 + ox_s), np.uint8)
        sh[oy_s:, ox_s:] = tag_bin * 255
        sh = sh[:H2, :W2]
        sh_f = cv2.GaussianBlur(sh.astype(np.float32), (15, 15), 0) / 255.0
        img = np.clip(img.astype(np.float32) * (1 - sh_f[:,:,None] * 0.22), 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), msk


scenes = sorted(IMG_DIR.glob('*.jpg'))
total  = len(scenes) * N_AUG
done   = 0
t0     = time.time()

for img_path in scenes:
    stem    = img_path.stem
    msk_path = MSK_DIR / (stem + '.png')
    lbl_path = LBL_DIR / (stem + '.txt')
    if not msk_path.exists():
        continue

    img_bgr = cv2.imread(str(img_path))
    msk_rgb = np.array(Image.open(msk_path))
    lbl_txt = lbl_path.read_text() if lbl_path.exists() else ''

    for ai in range(N_AUG):
        seed = int(time.time() * 1000 + done) % (2**31)
        aug_img, aug_msk = augment_scene(img_bgr, msk_rgb, seed)

        out_stem = f'{stem}_a{ai}'
        cv2.imwrite(str(IMG_DIR / f'{out_stem}.jpg'), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        Image.fromarray(aug_msk).save(str(MSK_DIR / f'{out_stem}.png'))
        H_aug, W_aug = aug_msk.shape[:2]
        lines = mask_to_yolo(aug_msk, W_aug, H_aug)
        (LBL_DIR / f'{out_stem}.txt').write_text('\n'.join(lines))

        done += 1
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        remain  = (total - done) / rate if rate > 0 else 0
        print(f'{done}/{total}  kalan: {remain/60:.1f} dk', end='\r')

print(f'\nBitti! {done} augmented sahne uretildi.')
