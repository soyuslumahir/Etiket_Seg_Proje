import json, random, numpy as np, cv2
from pathlib import Path
from PIL import Image

from gen_sample import render_pair

def randomize_pin_shape(img_pil, msk_pil, cfg):
    pad = cfg['top_pad']
    cx  = 785 // 2
    pcy = cfg['pin_cy'] + pad
    pr  = cfg['pin_r']
    if pr < 5:
        return img_pil, msk_pil

    lc = tuple(int(cfg['label_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bg = tuple(int(cfg['bg_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    img_arr = np.array(img_pil).copy()
    msk_arr = np.array(msk_pil).copy()

    erase = np.zeros(img_arr.shape[:2], dtype=np.uint8)
    cv2.circle(erase, (cx, pcy), pr + 4, 255, -1)
    erase[pcy:] = 0
    img_arr[erase > 0] = bg
    msk_arr[erase > 0] = (0, 0, 0)

    cy_shape = pcy - pr // 2

    shape = random.choice([
        'zigzag', 'ellipse', 'polygon', 'blob',
        'star', 'teardrop', 'diamond', 'squish',
        'kidney', 'spike', 'rect', 'wide'
    ])

    if shape == 'zigzag':
        n = random.randint(14, 24)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        lo, hi = pr * 0.5, pr * 1.1
        radii  = np.random.uniform(lo, hi, n)
        pts = np.array([[int(cx + r*np.cos(a)), int(cy_shape + r*np.sin(a))]
                        for r, a in zip(radii, angles)], dtype=np.int32)

    elif shape == 'ellipse':
        rx = random.randint(int(pr*0.5), int(pr*1.2))
        ry = random.randint(int(pr*0.5), int(pr*1.2))
        rot = random.randint(0, 180)
        pts = cv2.ellipse2Poly((cx, cy_shape), (rx, ry), rot, 0, 360, 8)

    elif shape == 'polygon':
        n = random.randint(4, 9)
        base_angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(pr * 0.6, pr * 1.1, n)
        pts = np.array([[int(cx + r*np.cos(a)), int(cy_shape + r*np.sin(a))]
                        for r, a in zip(radii, base_angles)], dtype=np.int32)

    elif shape == 'blob':
        n = 48
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii  = np.full(n, pr * 0.88)
        for freq in range(2, 7):
            amp   = random.uniform(0, pr * 0.20)
            phase = random.uniform(0, 2*np.pi)
            radii += amp * np.cos(freq * angles + phase)
        pts = np.array([[int(cx + r*np.cos(a)), int(cy_shape + r*np.sin(a))]
                        for r, a in zip(radii, angles)], dtype=np.int32)

    elif shape == 'star':
        n_spikes = random.randint(4, 8)
        angles = np.linspace(0, 2*np.pi, n_spikes * 2, endpoint=False)
        r_out = pr * random.uniform(0.85, 1.1)
        r_in  = pr * random.uniform(0.35, 0.60)
        radii = [r_out if i % 2 == 0 else r_in for i in range(n_spikes * 2)]
        pts = np.array([[int(cx + r*np.cos(a)), int(cy_shape + r*np.sin(a))]
                        for r, a in zip(radii, angles)], dtype=np.int32)

    elif shape == 'teardrop':
        # Uzun damla: uste sivri, alta yuvarlak
        n = 40
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        stretch = random.uniform(1.3, 2.0)
        radii = []
        for a in angles:
            r = pr * (0.7 + 0.3 * np.cos(a - np.pi/2))  # uste sivri
            radii.append(r)
        pts = np.array([[int(cx + r*np.cos(a)),
                         int(cy_shape + r*np.sin(a) * stretch)]
                        for r, a in zip(radii, angles)], dtype=np.int32)

    elif shape == 'diamond':
        s = pr * random.uniform(0.75, 1.05)
        pts = np.array([
            [cx, int(cy_shape - s)],
            [int(cx + s * random.uniform(0.6, 1.0)), cy_shape],
            [cx, int(cy_shape + s * random.uniform(0.5, 0.8))],
            [int(cx - s * random.uniform(0.6, 1.0)), cy_shape],
        ], dtype=np.int32)

    elif shape == 'squish':
        # Ezilmis yuvarlak: cok genis kisa
        rx = int(pr * random.uniform(1.1, 1.6))
        ry = int(pr * random.uniform(0.45, 0.70))
        pts = cv2.ellipse2Poly((cx, cy_shape), (rx, ry), 0, 0, 360, 6)

    elif shape == 'kidney':
        # Fasulye sekli: bir tarafi icbukeydi
        n = 48
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = []
        indent_phase = random.uniform(0, 2*np.pi)
        for a in angles:
            r = pr * (0.85 - 0.30 * max(0, np.cos(a - indent_phase)))
            radii.append(r)
        pts = np.array([[int(cx + r*np.cos(a)), int(cy_shape + r*np.sin(a))]
                        for r, a in zip(radii, angles)], dtype=np.int32)

    elif shape == 'spike':
        # Dikgen + tepede sivri spike
        n = random.randint(3, 5)
        angles = np.linspace(0, 2*np.pi, n * 2, endpoint=False)
        r_base = pr * random.uniform(0.55, 0.75)
        r_spike = pr * random.uniform(1.1, 1.5)
        radii = [r_spike if i % 2 == 0 else r_base for i in range(n * 2)]
        pts = np.array([[int(cx + r*np.cos(a)), int(cy_shape + r*np.sin(a))]
                        for r, a in zip(radii, angles)], dtype=np.int32)

    elif shape == 'rect':
        # Kose yuvarlatilmis dikdortgen
        rw = int(pr * random.uniform(0.7, 1.1))
        rh = int(pr * random.uniform(0.7, 1.1))
        pts = cv2.ellipse2Poly((cx, cy_shape), (rw, rh), 0, 0, 360, 30)

    else:  # wide — cok genis yayvan daire
        rx = int(pr * random.uniform(1.2, 1.7))
        ry = int(pr * random.uniform(0.7, 1.0))
        pts = cv2.ellipse2Poly((cx, cy_shape), (rx, ry), random.randint(0,30), 0, 360, 8)

    cv2.fillPoly(img_arr, [pts], lc)
    cv2.fillPoly(msk_arr, [pts], (200, 200, 200))

    # Shaft baglantiyi koru: yeni sekil ile pcy arasi bos kalabilir
    sh = cfg['shaft_w'] // 2
    cv2.rectangle(img_arr, (cx - sh, cy_shape), (cx + sh, pcy), lc, -1)
    cv2.rectangle(msk_arr, (cx - sh, cy_shape), (cx + sh, pcy), (200, 200, 200), -1)

    return Image.fromarray(img_arr), Image.fromarray(msk_arr)


cfg_path = Path('configs/tip2_0005.json')
with open(cfg_path) as f:
    cfg = json.load(f)

img_orig, msk_orig = render_pair(cfg, show_qr=True, number=cfg.get('number', '12345678'))

n_variants = 11
W_e, H_e = img_orig.width, img_orig.height
pad_px = 6

# Her satir: gorsel | mask yan yana, her variant bir satir
canvas = Image.new('RGB', (2*(W_e+pad_px)+pad_px, (n_variants+1)*(H_e+pad_px)+pad_px), (40,40,40))

def paste_row(row, img, msk):
    y = pad_px + row * (H_e + pad_px)
    canvas.paste(img, (pad_px, y))
    # Mask'i gorsel olarak goster (gri tona cevir, biraz parlat)
    msk_vis = Image.fromarray(np.clip(np.array(msk).astype(np.int32) * 1, 0, 255).astype(np.uint8))
    canvas.paste(msk_vis, (pad_px + W_e + pad_px, y))

paste_row(0, img_orig, msk_orig)

for i in range(1, n_variants + 1):
    random.seed(i * 17)
    np.random.seed(i * 17)
    img_out, msk_out = randomize_pin_shape(img_orig, msk_orig, cfg)
    paste_row(i, img_out, msk_out)

canvas.save('test_pin_shape_mask.png')
print('Kaydedildi: test_pin_shape_mask.png  (sol=gorsel, sag=mask)')
