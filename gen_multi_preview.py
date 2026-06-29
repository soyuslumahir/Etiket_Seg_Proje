import json, random, cv2, numpy as np
from pathlib import Path
from PIL import Image
from gen_sample import render_pair

import time
random.seed(int(time.time()))
np.random.seed(int(time.time()) % (2**32))

CONFIGS_DIR = Path('configs')
configs = sorted(CONFIGS_DIR.glob('*.json'))

BG_COLOR = (209, 196, 38)   # sari (RGB)
CANVAS_W, CANVAS_H = 4000, 3000
N_ETIKET = 12

canvas     = Image.new('RGB', (CANVAS_W, CANVAS_H), BG_COLOR)
mask_canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), (0, 0, 0))

cols, rows = 4, 3
cell_w = CANVAS_W // cols
cell_h = CANVAS_H // rows

for i in range(N_ETIKET):
    cfg_path = random.choice(configs)
    with open(cfg_path) as f:
        cfg = json.load(f)
    show_qr = 'tip2' in cfg_path.stem
    number  = cfg.get('number', '12345678')
    angle   = random.uniform(-20, 20)
    scale   = random.uniform(0.60, 0.80)

    img_pil, mask_pil = render_pair(cfg, show_qr=show_qr, number=number)

    # Maske piksellerinden alpha olustur
    mask_np = np.array(mask_pil)
    alpha   = (mask_np.max(axis=2) > 10).astype(np.uint8) * 255

    # Gorsel RGBA
    img_rgba = img_pil.convert('RGBA')
    r, g, b, _ = img_rgba.split()
    img_rgba = Image.merge('RGBA', (r, g, b, Image.fromarray(alpha)))

    # Maske RGBA (siyah arka plan, renkli siniflar)
    msk_rgba = mask_pil.convert('RGBA')
    mr, mg, mb, _ = msk_rgba.split()
    msk_rgba = Image.merge('RGBA', (mr, mg, mb, Image.fromarray(alpha)))

    # Kucult
    new_w = int(img_pil.width * scale)
    new_h = int(img_pil.height * scale)
    img_rgba = img_rgba.resize((new_w, new_h), Image.LANCZOS)
    msk_rgba = msk_rgba.resize((new_w, new_h), Image.NEAREST)

    # Dondur
    img_rot = img_rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
    msk_rot = msk_rgba.rotate(angle, expand=True, resample=Image.NEAREST)

    # Grid konumu
    col = i % cols
    row = i // cols
    cx  = col * cell_w + cell_w // 2
    cy  = row * cell_h + cell_h // 2
    ox  = random.randint(-cell_w // 6, cell_w // 6)
    oy  = random.randint(-cell_h // 6, cell_h // 6)
    px  = cx - img_rot.width  // 2 + ox
    py  = cy - img_rot.height // 2 + oy

    canvas.paste(img_rot,  (px, py), mask=img_rot.split()[3])
    mask_canvas.paste(msk_rot, (px, py), mask=msk_rot.split()[3])
    print(f"  Etiket {i+1}: {cfg_path.name}  aci={angle:.1f}  olcek={scale:.2f}")

canvas.save('multi_preview.jpg', quality=90)
mask_canvas.save('multi_preview_mask.png')   # PNG: kayipsiz, kenar bozulmasi yok
print(f"\nKaydedildi: multi_preview.jpg + multi_preview_mask.png  ({CANVAS_W}x{CANVAS_H})")
