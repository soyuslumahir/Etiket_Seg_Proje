import json, random, cv2, numpy as np, time, math
from pathlib import Path
from PIL import Image

random.seed(int(time.time()))
np.random.seed(int(time.time()) % (2**32))

from gen_sample import render_pair

CONFIGS_DIR = Path('configs')
OUT_DIR     = Path('scenes')
OUT_IMG     = OUT_DIR / 'images'
OUT_MSK     = OUT_DIR / 'masks'
OUT_LBL     = OUT_DIR / 'labels'
for d in [OUT_IMG, OUT_MSK, OUT_LBL]:
    d.mkdir(parents=True, exist_ok=True)

BG_COLOR   = (209, 196, 38)
CANVAS_W   = 4000
CANVAS_H   = 3000
PLACE_X0   = 500    # merkez 3000x3000: x 500–3500
PLACE_X1   = 3500
PLACE_Y0   = 200
PLACE_Y1   = 2800
PLACE_W    = PLACE_X1 - PLACE_X0   # 3000
PLACE_H    = PLACE_Y1 - PLACE_Y0   # 3000
N_SCENES   = 101
SCENE_OFFSET = 399
FILL       = 0.97   # hucre doldurma orani

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# Etiket ham boyutlari (etiket_editor: W=785, H~922)
ETK_W, ETK_H = 785, 922

CC = {
    0:(200,200,200), 1:(0,200,0),   2:(0,100,255),  3:(255,165,0),
    4:(255,50,50),   5:(255,150,50), 6:(255,230,50), 7:(50,255,50),
    8:(50,255,200),  9:(50,200,255), 10:(50,50,255), 11:(200,50,255),
    12:(255,50,200), 13:(180,180,100)
}

configs = sorted(CONFIGS_DIR.glob('*.json'))


def randomize_pin_shape(img_pil, msk_pil, cfg):
    pad = cfg['top_pad']
    cx  = 785 // 2
    pcy = cfg['pin_cy'] + pad
    pr  = cfg['pin_r']
    if pr < 5 or random.random() < 0.50:
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

    cy_s = pcy - pr // 2

    shape = random.choice([
        'zigzag', 'ellipse', 'polygon', 'blob',
        'star', 'teardrop', 'diamond', 'squish',
        'kidney', 'spike', 'wide', 'circle'
    ])

    if shape == 'zigzag':
        n = random.randint(14, 24)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii  = np.random.uniform(pr*0.5, pr*1.1, n)
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a))] for r,a in zip(radii,angles)], dtype=np.int32)
    elif shape == 'ellipse':
        rx  = random.randint(int(pr*0.5), int(pr*1.2))
        ry  = random.randint(int(pr*0.5), int(pr*1.2))
        pts = cv2.ellipse2Poly((cx, cy_s), (rx, ry), random.randint(0,180), 0, 360, 8)
    elif shape == 'polygon':
        n = random.randint(4, 9)
        ba = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(pr*0.6, pr*1.1, n)
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a))] for r,a in zip(radii,ba)], dtype=np.int32)
    elif shape == 'blob':
        n = 48; angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = np.full(n, pr*0.88)
        for freq in range(2, 7):
            radii += random.uniform(0, pr*0.20) * np.cos(freq*angles + random.uniform(0, 2*np.pi))
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a))] for r,a in zip(radii,angles)], dtype=np.int32)
    elif shape == 'star':
        ns = random.randint(4, 8)
        angles = np.linspace(0, 2*np.pi, ns*2, endpoint=False)
        radii  = [pr*random.uniform(0.85,1.1) if i%2==0 else pr*random.uniform(0.35,0.60) for i in range(ns*2)]
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a))] for r,a in zip(radii,angles)], dtype=np.int32)
    elif shape == 'teardrop':
        n = 40; angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        stretch = random.uniform(1.3, 2.0)
        radii = [pr*(0.7+0.3*np.cos(a-np.pi/2)) for a in angles]
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a)*stretch)] for r,a in zip(radii,angles)], dtype=np.int32)
    elif shape == 'diamond':
        s = pr*random.uniform(0.75, 1.05)
        pts = np.array([[cx,int(cy_s-s)],[int(cx+s*random.uniform(0.6,1.0)),cy_s],
                        [cx,int(cy_s+s*random.uniform(0.5,0.8))],[int(cx-s*random.uniform(0.6,1.0)),cy_s]], dtype=np.int32)
    elif shape == 'squish':
        pts = cv2.ellipse2Poly((cx, cy_s), (int(pr*random.uniform(1.1,1.6)), int(pr*random.uniform(0.45,0.70))), 0, 0, 360, 6)
    elif shape == 'kidney':
        n = 48; angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        ip = random.uniform(0, 2*np.pi)
        radii = [pr*(0.85-0.30*max(0, np.cos(a-ip))) for a in angles]
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a))] for r,a in zip(radii,angles)], dtype=np.int32)
    elif shape == 'spike':
        ns = random.randint(3, 5)
        angles = np.linspace(0, 2*np.pi, ns*2, endpoint=False)
        radii  = [pr*random.uniform(1.1,1.5) if i%2==0 else pr*random.uniform(0.55,0.75) for i in range(ns*2)]
        pts = np.array([[int(cx+r*np.cos(a)), int(cy_s+r*np.sin(a))] for r,a in zip(radii,angles)], dtype=np.int32)
    elif shape == 'wide':
        pts = cv2.ellipse2Poly((cx, cy_s), (int(pr*random.uniform(1.2,1.7)), int(pr*random.uniform(0.7,1.0))), random.randint(0,30), 0, 360, 8)
    else:  # circle — orijinale benzer ama boyut degisik
        r2 = int(pr * random.uniform(0.7, 1.15))
        pts = cv2.ellipse2Poly((cx, cy_s), (r2, r2), 0, 0, 360, 8)

    cv2.fillPoly(img_arr, [pts], lc)
    cv2.fillPoly(msk_arr, [pts], (200, 200, 200))

    sh = cfg['shaft_w'] // 2
    cv2.rectangle(img_arr, (cx-sh, cy_s), (cx+sh, pcy), lc, -1)
    cv2.rectangle(msk_arr, (cx-sh, cy_s), (cx+sh, pcy), (200, 200, 200), -1)

    return Image.fromarray(img_arr), Image.fromarray(msk_arr)


def rotated_bbox(w, h, deg):
    rad = math.radians(deg)
    return (abs(w*math.cos(rad)) + abs(h*math.sin(rad)),
            abs(w*math.sin(rad)) + abs(h*math.cos(rad)))



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


for si in range(N_SCENES):
    cols = random.randint(3, 4)
    rows = random.randint(3, 4)
    n_etiket = cols * rows   # tum hucreler dolu
    cell_w = PLACE_W / cols
    cell_h = PLACE_H / rows

    cells = [(c, r) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)

    canvas      = Image.new('RGB', (CANVAS_W, CANVAS_H), BG_COLOR)
    mask_canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), (0, 0, 0))

    for (col, row) in cells:
        cfg_path = random.choice(configs)
        with open(cfg_path) as f:
            cfg = json.load(f)
        show_qr = 'tip2' in cfg_path.stem
        number  = ''.join([str(random.randint(0, 9)) for _ in range(8)])

        img_pil, msk_pil = render_pair(cfg, show_qr=show_qr, number=number)
        img_pil, msk_pil = randomize_pin_shape(img_pil, msk_pil, cfg)

        angle = random.choice(ANGLES)

        # Bu aci icin scale: etiket hucreye FILL oraninda sigacak sekilde
        bbox_w, bbox_h = rotated_bbox(ETK_W, ETK_H, angle)
        scale = min(cell_w / bbox_w, cell_h / bbox_h) * FILL

        new_w = int(img_pil.width  * scale)
        new_h = int(img_pil.height * scale)

        msk_np = np.array(msk_pil)
        alpha  = Image.fromarray((msk_np.max(axis=2) > 10).astype(np.uint8) * 255)

        img_rgba = img_pil.convert('RGBA')
        r, g, b, _ = img_rgba.split()
        img_rgba = Image.merge('RGBA', (r, g, b, alpha))

        msk_rgba = msk_pil.convert('RGBA')
        mr, mg, mb, _ = msk_rgba.split()
        msk_rgba = Image.merge('RGBA', (mr, mg, mb, alpha))

        img_rgba = img_rgba.resize((new_w, new_h), Image.LANCZOS)
        msk_rgba = msk_rgba.resize((new_w, new_h), Image.NEAREST)

        img_rot = img_rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
        msk_rot = msk_rgba.rotate(angle, expand=True, resample=Image.NEAREST)

        rw, rh = img_rot.size

        # Hucre merkezi + kucuk rastgele offset (hucrenin %10u kadar)
        cx = PLACE_X0 + (col + 0.5) * cell_w
        cy = PLACE_Y0 + (row + 0.5) * cell_h
        ox = random.uniform(-cell_w * 0.03, cell_w * 0.03)
        oy = random.uniform(-cell_h * 0.03, cell_h * 0.03)
        px = int(cx - rw / 2 + ox)
        py = int(cy - rh / 2 + oy)

        # Canvas siniri icinde tut
        px = max(0, min(CANVAS_W - rw, px))
        py = max(0, min(CANVAS_H - rh, py))

        canvas.paste(img_rot,  (px, py), mask=img_rot.split()[3])
        mask_canvas.paste(msk_rot, (px, py), mask=msk_rot.split()[3])

    stem = f'scene_{si+SCENE_OFFSET:04d}'
    canvas.save(str(OUT_IMG / f'{stem}.jpg'), quality=92)
    mask_canvas.save(str(OUT_MSK / f'{stem}.png'))

    msk_np = np.array(mask_canvas)
    lines  = mask_to_yolo(msk_np, CANVAS_W, CANVAS_H)
    (OUT_LBL / f'{stem}.txt').write_text('\n'.join(lines))

    print(f'Sahne {si+1}/{N_SCENES}: {n_etiket} etiket  grid={cols}x{rows}  {len(lines)} annotation')

print('Bitti!')
