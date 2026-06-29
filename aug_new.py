import cv2, numpy as np, random, json
from pathlib import Path
import albumentations as A

# ------------------------------------------------------------------ #
#  Tek bir (img_np BGR, mask_np RGB) cifti augmente eder             #
#  Geometrik transformlar her ikisine de uygulanir                   #
# ------------------------------------------------------------------ #

BG_COLOR_RGB = (209, 196, 38)   # sari arka plan (#d1c426)


def augment(img_bgr: np.ndarray, mask_rgb: np.ndarray,
            seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.RandomState(seed if seed is not None else random.randint(0, 99999))

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    msk = mask_rgb.copy()

    # Rotation sırasında kesme olmasin diye onceden pad ekle
    H0, W0 = img.shape[:2]
    pad = int(max(H0, W0) * 0.20)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=BG_COLOR_RGB)
    msk = cv2.copyMakeBorder(msk, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # ============================================================
    # STAGE 1 — Geometric (image + mask)
    # ============================================================
    # Discrete rotation: 0, 45, 90, 135, 180, 225, 270, 315
    angle = int(rng.choice([0, 45, 90, 135, 180, 225, 270, 315]))
    if angle != 0:
        Hp, Wp = img.shape[:2]
        Mc = cv2.getRotationMatrix2D((Wp // 2, Hp // 2), angle, 1.0)
        img = cv2.warpAffine(img, Mc, (Wp, Hp), borderMode=cv2.BORDER_CONSTANT,
                             borderValue=BG_COLOR_RGB)
        msk = cv2.warpAffine(msk, Mc, (Wp, Hp), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    stage1 = A.Compose([
        A.Perspective(scale=(0.02, 0.05), pad_mode=cv2.BORDER_CONSTANT, pad_val=BG_COLOR_RGB, p=0.5),
        A.Affine(shear=(-3, 3), scale=(0.97, 1.03), mode=cv2.BORDER_CONSTANT, cval=BG_COLOR_RGB, p=0.5),
    ], additional_targets={'mask': 'mask'})
    r = stage1(image=img, mask=msk)
    img, msk = r['image'], r['mask']

    # ============================================================
    # STAGE 2 — Scale: camera distance (image + mask)
    # ============================================================
    H0, W0 = img.shape[:2]
    scale_pct = rng.uniform(0.60, 0.90)
    tag_px = int(max(H0, W0) * scale_pct)
    img_s = cv2.resize(img, (tag_px, tag_px))
    msk_s = cv2.resize(msk, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)

    canvas_sz = max(H0, W0)
    canvas = np.full((canvas_sz, canvas_sz, 3), BG_COLOR_RGB, dtype=np.uint8)
    canvas_m = np.zeros((canvas_sz, canvas_sz, 3), dtype=np.uint8)
    ox = (canvas_sz - tag_px) // 2 + rng.randint(-8, 8)
    oy = (canvas_sz - tag_px) // 2 + rng.randint(-8, 8)
    ox = int(np.clip(ox, 0, canvas_sz - tag_px))
    oy = int(np.clip(oy, 0, canvas_sz - tag_px))
    canvas[oy:oy+tag_px, ox:ox+tag_px] = img_s
    canvas_m[oy:oy+tag_px, ox:ox+tag_px] = msk_s
    img, msk = canvas, canvas_m

    # ============================================================
    # STAGE 3 — Lighting & color (image only)
    # ============================================================
    stage3 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.7),
        A.HueSaturationValue(hue_shift_limit=7, sat_shift_limit=18, val_shift_limit=12, p=0.7),
        A.RandomGamma(gamma_limit=(85, 120), p=0.5),
    ])
    img = stage3(image=img)['image']

    # Yellow cast
    if rng.random() < 0.65:
        yellow = np.full_like(img, (220, 200, 50), dtype=np.uint8)
        a = rng.uniform(0.03, 0.09)
        img = cv2.addWeighted(img, 1 - a, yellow, a, 0)

    # ============================================================
    # STAGE 4 — Surface & noise (image only)
    # ============================================================
    stage4 = A.Compose([
        A.GaussNoise(var_limit=(3.0, 12.0), p=0.6),
        A.MotionBlur(blur_limit=(3, 5), p=0.15),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),
    ])
    img = stage4(image=img)['image']

    # Specular highlights (image only)
    tag_any = msk.max(axis=2) > 10
    ys, xs = np.where(tag_any)
    if len(xs) > 0:
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, len(xs))
            spec = np.zeros_like(img)
            rx, ry = rng.randint(8, 25), rng.randint(8, 25)
            cv2.ellipse(spec, (int(xs[idx]), int(ys[idx])), (rx, ry),
                        int(rng.uniform(0, 360)), 0, 360, (255, 255, 255), -1)
            spec = cv2.GaussianBlur(spec, (21, 21), 0)
            img = cv2.addWeighted(img, 1.0, spec, rng.uniform(0.05, 0.16), 0)

    # ============================================================
    # STAGE 5 — Text fading (image only, tag region)
    # ============================================================
    if len(xs) > 0:
        x1t, x2t = int(xs.min()), int(xs.max())
        y1t, y2t = int(ys.min()), int(ys.max())
        if x2t > x1t and y2t > y1t:
            roi = img[y1t:y2t, x1t:x2t]
            # Slight desaturation
            if rng.random() < 0.5:
                hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= rng.uniform(0.83, 0.97)
                roi = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            # Hue shift (dark gray / dark blue / near-black tones)
            if rng.random() < 0.5:
                hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
                roi = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            img[y1t:y2t, x1t:x2t] = roi

    # ============================================================
    # STAGE 6 — Physical deformation (image + mask)
    # ============================================================
    stage6 = A.Compose([
        A.ElasticTransform(alpha=15, sigma=4, alpha_affine=0, p=0.4),
    ], additional_targets={'mask': 'mask'})
    r = stage6(image=img, mask=msk)
    img, msk = r['image'], r['mask']

    # Pin hole near top-center (image only)
    tag_any2 = msk.max(axis=2) > 10
    ys2, xs2 = np.where(tag_any2)
    if len(xs2) > 0 and rng.random() < 0.45:
        px = int((xs2.min() + xs2.max()) / 2) + rng.randint(-15, 15)
        py = int(ys2.min()) + rng.randint(4, 18)
        pr = rng.randint(3, 7)
        cv2.circle(img, (px, py), pr, (15, 15, 15), -1)

    # Edge crease/vignette (image only)
    if rng.random() < 0.6:
        H2, W2 = img.shape[:2]
        edge = rng.choice(['top', 'bottom', 'left', 'right'])
        dark = np.zeros((H2, W2), np.float32)
        t = rng.randint(20, 55)
        s = rng.uniform(0.07, 0.22)
        if   edge == 'top':    dark[:t, :] = np.linspace(s, 0, t)[:, None]
        elif edge == 'bottom': dark[-t:, :] = np.linspace(0, s, t)[:, None]
        elif edge == 'left':   dark[:, :t] = np.linspace(s, 0, t)[None, :]
        else:                  dark[:, -t:] = np.linspace(0, s, t)[None, :]
        dark = cv2.GaussianBlur(dark, (31, 31), 0)
        img = np.clip(img.astype(np.float32) * (1 - dark[:, :, None]), 0, 255).astype(np.uint8)

    # ============================================================
    # STAGE 7 — Compositing: shadow (image only)
    # ============================================================
    if rng.random() < 0.65:
        H2, W2 = img.shape[:2]
        tag_bin = (msk.max(axis=2) > 10).astype(np.uint8)
        ox_s, oy_s = rng.randint(3, 9), rng.randint(3, 9)
        sh = np.zeros((H2 + oy_s, W2 + ox_s), np.uint8)
        sh[oy_s:, ox_s:] = tag_bin * 255
        sh = sh[:H2, :W2]
        sh_f = cv2.GaussianBlur(sh.astype(np.float32), (7, 7), 0) / 255.0
        img_f = img.astype(np.float32) * (1 - sh_f[:, :, None] * 0.22)
        img = np.clip(img_f, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), msk


# ------------------------------------------------------------------ #
#  Preview: 10-20 ornek grid                                          #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from gen_sample import render_pair

    configs = sorted(Path('configs').glob('*.json'))
    chosen = random.sample(configs, 15)

    THUMB = 400
    cols, rows = 5, 3
    canvas_out = np.zeros((rows * THUMB, cols * THUMB, 3), np.uint8)

    for idx, cfg_path in enumerate(chosen):
        with open(cfg_path) as f:
            cfg = json.load(f)
        number = cfg.get('number', '12345678')
        show_qr = 'tip2' in cfg_path.name
        img_pil, msk_pil = render_pair(cfg, show_qr=show_qr, number=number)

        import cv2 as _cv2
        img_bgr = _cv2.cvtColor(np.array(img_pil), _cv2.COLOR_RGB2BGR)
        msk_rgb = np.array(msk_pil)

        aug_bgr, _ = augment(img_bgr, msk_rgb, seed=idx * 7)
        thumb = _cv2.resize(aug_bgr, (THUMB, THUMB))
        r, c = divmod(idx, cols)
        canvas_out[r*THUMB:(r+1)*THUMB, c*THUMB:(c+1)*THUMB] = thumb

    out_path = r'C:\Users\soyus\Desktop\aug_new_preview.jpg'
    cv2.imwrite(out_path, canvas_out)
    print('Kaydedildi:', out_path)
