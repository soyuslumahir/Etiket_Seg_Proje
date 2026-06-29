import cv2, numpy as np, random
from PIL import Image
import albumentations as A

random.seed(42)
np.random.seed(42)
rng = np.random.RandomState(42)

img_bgr = cv2.imread('scenes/images/scene_0050.jpg')
msk_rgb = np.array(Image.open('scenes/masks/scene_0050.png'))

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
msk = msk_rgb.copy()

# Sahne rotasyonu (discrete)
angle = int(rng.choice([0, 90, 180, 270]))
if angle != 0:
    H0, W0 = img.shape[:2]
    Mc = cv2.getRotationMatrix2D((W0//2, H0//2), angle, 1.0)
    bg_rgb = (209, 196, 38)
    img = cv2.warpAffine(img, Mc, (W0, H0), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_rgb)
    msk = cv2.warpAffine(msk, Mc, (W0, H0), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

# Stage 3 — Lighting & color
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

# Stage 4 — Noise
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

# Stage 7 — Shadow
if rng.random() < 0.65:
    H2, W2 = img.shape[:2]
    tag_bin = (msk.max(axis=2) > 10).astype(np.uint8)
    ox_s, oy_s = rng.randint(5, 18), rng.randint(5, 18)
    sh = np.zeros((H2 + oy_s, W2 + ox_s), np.uint8)
    sh[oy_s:, ox_s:] = tag_bin * 255
    sh = sh[:H2, :W2]
    sh_f = cv2.GaussianBlur(sh.astype(np.float32), (15, 15), 0) / 255.0
    img_f = img.astype(np.float32) * (1 - sh_f[:, :, None] * 0.22)
    img = np.clip(img_f, 0, 255).astype(np.uint8)

aug_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Yan yana karsilastirma (kucultulmus)
scale = 3
h, w = img_bgr.shape[:2]
orig_s = cv2.resize(img_bgr, (w//scale, h//scale))
aug_s  = cv2.resize(aug_bgr, (w//scale, h//scale))
comp = np.hstack([orig_s, np.full((h//scale, 10, 3), 40, np.uint8), aug_s])
cv2.imwrite('test_aug_scene.jpg', comp, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('Kaydedildi: test_aug_scene.jpg')
