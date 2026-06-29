import json, random, sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, '.')
from gen_sample import render_pair
from aug_new import augment

CONFIGS_DIR = Path('configs')
OUT_IMG = Path('dataset_aug/images')
OUT_MSK = Path('dataset_aug/masks')
OUT_LBL = Path('dataset_aug/labels')

CC = {0:(200,200,200),1:(0,200,0),2:(0,100,255),3:(255,165,0),4:(255,50,50),
      5:(255,150,50),6:(255,230,50),7:(50,255,50),8:(50,255,200),9:(50,200,255),
      10:(50,50,255),11:(200,50,255),12:(255,50,200),13:(180,180,100)}

N_AUG = 2   # her config icin 2 augmentation → 1000 × 2 = 2000

def mask_to_yolo(msk_rgb, W, H):
    lines = []
    for cid, color in CC.items():
        if cid == 0:
            # Etiket: tum non-black piksellerin dis konturunu al
            # (digit renkleri etiket piksellerini silip adalar olusturur)
            binary = (msk_rgb.max(axis=2) > 10).astype(np.uint8) * 255
            min_area = 1000
        else:
            c = np.array(color, dtype=np.int32)
            diff = np.abs(msk_rgb.astype(np.int32) - c).max(axis=2)
            binary = (diff < 5).astype(np.uint8) * 255
            min_area = 50
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

configs = sorted(CONFIGS_DIR.glob('*.json'))
total = len(configs) * N_AUG
done = 0

for cfg_path in configs:
    with open(cfg_path) as f:
        cfg = json.load(f)
    number = cfg.get('number', '12345678')
    show_qr = 'tip2' in cfg_path.name

    img_pil, msk_pil = render_pair(cfg, show_qr=show_qr, number=number)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    msk_rgb = np.array(msk_pil)

    for ai in range(N_AUG):
        seed = abs(hash(cfg_path.name)) % (2**31) + ai * 9973
        stem = f'{cfg_path.stem}_a{ai}'

        aug_bgr, aug_msk = augment(img_bgr, msk_rgb, seed=seed)

        cv2.imwrite(str(OUT_IMG / f'{stem}.jpg'), aug_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT_MSK / f'{stem}.png'),
                    cv2.cvtColor(aug_msk, cv2.COLOR_RGB2BGR))

        H, W = aug_bgr.shape[:2]
        lines = mask_to_yolo(aug_msk, W, H)
        (OUT_LBL / f'{stem}.txt').write_text('\n'.join(lines))

        done += 1
        if done % 200 == 0:
            print(f'{done}/{total}', flush=True)

print(f'Bitti! {done} goruntu uretildi.')
