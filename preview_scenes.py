import cv2, numpy as np
from pathlib import Path
from PIL import Image

OUT_DIR = Path('scenes')
IMG_DIR = OUT_DIR / 'images'
MSK_DIR = OUT_DIR / 'masks'
PREV_DIR = OUT_DIR / 'previews'
PREV_DIR.mkdir(exist_ok=True)

CC = {
    0:(200,200,200), 1:(0,200,0),   2:(0,100,255),  3:(255,165,0),
    4:(255,50,50),   5:(255,150,50), 6:(255,230,50), 7:(50,255,50),
    8:(50,255,200),  9:(50,200,255), 10:(50,50,255), 11:(200,50,255),
    12:(255,50,200), 13:(180,180,100)
}

for img_path in sorted(IMG_DIR.glob('*.jpg')):
    msk_path = MSK_DIR / (img_path.stem + '.png')
    if not msk_path.exists():
        continue

    img_bgr = cv2.imread(str(img_path))
    msk_rgb = np.array(Image.open(msk_path))

    # Yarim saydam mask overlay
    overlay = img_bgr.copy()
    for cid, color in CC.items():
        c = np.array(color, dtype=np.int32)
        diff = np.abs(msk_rgb.astype(np.int32) - c).max(axis=2)
        binary = (diff < 5).astype(np.uint8) * 255
        overlay[binary > 0] = (color[2], color[1], color[0])  # BGR

    blended = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)

    # Kontur ciz
    for cid, color in CC.items():
        c = np.array(color, dtype=np.int32)
        diff = np.abs(msk_rgb.astype(np.int32) - c).max(axis=2)
        binary = (diff < 5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 100:
                continue
            cv2.drawContours(blended, [cnt], -1, (color[2], color[1], color[0]), 3)

    # Kucult (4000x3000 → 1333x1000)
    h, w = blended.shape[:2]
    small = cv2.resize(blended, (w//3, h//3))

    out_path = PREV_DIR / (img_path.stem + '_preview.jpg')
    cv2.imwrite(str(out_path), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'{out_path.name}')

print('Bitti! previews/ klasorune bak.')
