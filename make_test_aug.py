import cv2
import numpy as np
from pathlib import Path

SRC_DIR = Path(r"C:\Users\soyus\Downloads\raw_images_4000x3000")
DST_DIR = Path(r"C:\Users\soyus\Desktop\etiket-proje\test_gercek_aug")
DST_DIR.mkdir(exist_ok=True)

IMGS = [
    "usb_frame_20260622_103722_033.jpg",
    "usb_frame_20260622_103815_642.jpg",
]

def adjust(img, brightness=0, contrast=1.0, hue_shift=0, sat_scale=1.0):
    out = img.astype(np.float32)
    out = np.clip(out * contrast + brightness, 0, 255).astype(np.uint8)
    if hue_shift != 0 or sat_scale != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_scale, 0, 255)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def rotate(img, angle):
    h, w = img.shape[:2]
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

# 8 augmentasyon tanimi (her foto icin ayni)
AUGS = [
    dict(angle=90,  brightness=0,    contrast=1.0,  hue_shift=0,  sat_scale=1.0),
    dict(angle=180, brightness=0,    contrast=1.0,  hue_shift=0,  sat_scale=1.0),
    dict(angle=270, brightness=0,    contrast=1.0,  hue_shift=0,  sat_scale=1.0),
    dict(angle=0,   brightness=30,   contrast=1.1,  hue_shift=0,  sat_scale=1.0),
    dict(angle=0,   brightness=-30,  contrast=0.9,  hue_shift=0,  sat_scale=1.0),
    dict(angle=0,   brightness=0,    contrast=1.2,  hue_shift=5,  sat_scale=1.1),
    dict(angle=90,  brightness=20,   contrast=1.1,  hue_shift=0,  sat_scale=1.0),
    dict(angle=180, brightness=-20,  contrast=0.9,  hue_shift=-5, sat_scale=0.9),
]

for img_name in IMGS:
    src = SRC_DIR / img_name
    img = cv2.imread(str(src))
    if img is None:
        print(f"BULUNAMADI: {src}")
        continue

    stem = Path(img_name).stem

    # Orijinal kaydet
    dst_orig = DST_DIR / img_name
    cv2.imwrite(str(dst_orig), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Orijinal: {dst_orig.name}")

    # 8 augmentasyon
    for i, aug in enumerate(AUGS, 1):
        out = adjust(img, aug['brightness'], aug['contrast'], aug['hue_shift'], aug['sat_scale'])
        if aug['angle'] != 0:
            out = rotate(out, aug['angle'])
        fname = f"{stem}_aug{i:02d}.jpg"
        cv2.imwrite(str(DST_DIR / fname), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  aug{i}: rot={aug['angle']}° bright={aug['brightness']:+} contrast={aug['contrast']}")

print(f"\nTamam! {DST_DIR} klasorune kaydedildi.")
print(f"Toplam: {len(list(DST_DIR.glob('*.jpg')))} dosya")
