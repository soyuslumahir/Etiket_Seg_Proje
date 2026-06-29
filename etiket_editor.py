import math
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W_IMG = 785
LOGO_PATH = Path(__file__).parent / 'IMG_3824.PNG'

FONT_CACHE = {}
_LOGO_CACHE = {}

def _fit_font(text, max_h, max_w=None, path='arial.ttf'):
    tmp = ImageDraw.Draw(Image.new('L', (2000, 2000)))
    for fs in range(500, 5, -1):
        f = ImageFont.truetype(path, fs)
        bb = tmp.textbbox((0, 0), text, font=f)
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        if h <= max_h and (max_w is None or w <= max_w):
            return f
    return ImageFont.truetype(path, 10)

def _fit_font_cached(text, max_h, max_w=None, path='arial.ttf'):
    key = (text, max_h, max_w, path)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = _fit_font(text, max_h, max_w, path)
    return FONT_CACHE[key]

def _bezier3(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)
    x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
    y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
    return list(zip(x.astype(int), y.astype(int)))

def _bezier2(p0, p1, p2, n=25):
    t = np.linspace(0, 1, n)
    x = (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
    y = (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
    return list(zip(x.astype(int), y.astype(int)))


def render(cfg, show_qr=True, number='12345678'):
    pad = cfg['top_pad']
    H   = 847 + pad

    bm  = cfg['body_margin']
    cr  = cfg['corner_r']
    tcr = cfg['top_corner_r']
    sh  = cfg['shaft_w'] // 2
    sy  = cfg['shoulder_y'] + pad
    sb  = cfg['shaft_bot'] + pad
    bb_ = cfg['body_bot'] + pad
    cx  = W_IMG // 2
    pcy = cfg['pin_cy'] + pad
    pr  = cfg['pin_r']

    bg_color    = tuple(int(cfg['bg_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    label_color = tuple(int(cfg['label_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    text_color  = tuple(int(cfg['text_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Canvas
    img = np.full((H, W_IMG, 3), bg_color[::-1], dtype=np.uint8)  # BGR

    # --- Etiket govdesi ---
    # Omuz bezier t_len: omuz yuksekligine orantili → daha yumusak gecis
    t_len = max(70, int((sy - sb) * 0.55))

    # Pin arc: sag tangent noktasindan (0°) sol tangent noktasina (180°) ust yari daire
    arc_pts = []
    for a in range(0, 181, 3):
        rad = math.radians(a)
        arc_pts.append((int(cx + pr * math.cos(rad)),
                        int(pcy - pr * math.sin(rad))))

    # Tek buyuk polygon: govde + shaft kenarlari + pin arc
    pts = []
    sol = (bm + tcr, sy)

    # Sol shaft tabanından omuz bezier → sol govde kosesi
    pts += _bezier3((cx-sh, sb), (cx-sh, sb+t_len), (sol[0]+t_len, sol[1]), sol)
    if tcr > 0:
        pts += _bezier2((bm+tcr, sy), (bm, sy), (bm, sy+tcr), n=25)
    pts.append((bm, bb_-cr))
    pts += _bezier2((bm, bb_-cr), (bm, bb_), (bm+cr, bb_), n=20)
    pts.append((W_IMG-bm-cr, bb_))
    pts += _bezier2((W_IMG-bm-cr, bb_), (W_IMG-bm, bb_), (W_IMG-bm, bb_-cr), n=20)
    pts.append((W_IMG-bm, sy+tcr if tcr > 0 else sy))
    if tcr > 0:
        pts += _bezier2((W_IMG-bm, sy+tcr), (W_IMG-bm, sy), (W_IMG-bm-tcr, sy), n=20)
    sag = (W_IMG-bm-tcr, sy)
    # Sag govde kosesinden omuz bezier → sag shaft tabani
    pts += _bezier3(sag, (sag[0]-t_len, sag[1]), (cx+sh, sb+t_len), (cx+sh, sb))
    # Sag shaft yukarı pcy'ye
    pts.append((cx+sh, pcy))
    # Pin arc (sag 0° → sol 180°, ust yari daire)
    pts += arc_pts
    # Sol shaft asagi sb'ye — polygon kapanacak

    pts_cv = np.array(pts, dtype=np.int32)
    outline_color = tuple(max(0, c-40) for c in label_color[::-1])
    cv2.fillPoly(img, [pts_cv], label_color[::-1])
    cv2.polylines(img, [pts_cv], True, outline_color, 2)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Ortak satir merkezi: TR/logo/QR ayni yatay cizgide
    row_cy = round((cfg['tr_cy'] + cfg['logo_cy'] + cfg['qr_cy']) / 3) + pad

    # --- TR yazisi ---
    ft = _fit_font_cached('TR', cfg['tr_h'])
    d_tmp = ImageDraw.Draw(Image.new('L', (1, 1)))
    bt = d_tmp.textbbox((0, 0), 'TR', font=ft)
    tx = cfg['tr_cx'] - (bt[2]-bt[0])//2
    ty = row_cy - (bt[3]-bt[1])//2 - bt[1]
    draw.text((tx, ty), 'TR', fill=text_color, font=ft)

    # --- Logo (gercek resim) ---
    lw = cfg['logo_w']
    lh = cfg['logo_h']
    lx = cfg['logo_cx'] - lw // 2
    ly = row_cy - lh // 2
    cache_key = (lw, lh, text_color, bg_color)
    if cache_key not in _LOGO_CACHE:
        raw = Image.open(LOGO_PATH).convert('RGBA')
        raw = raw.resize((lw, lh), Image.LANCZOS)
        arr = np.array(raw)
        design = arr[:, :, 3] > 0   # transparan olmayan pikseller = logo dizayni
        out = np.zeros((lh, lw, 4), dtype=np.uint8)
        out[design]  = [text_color[0], text_color[1], text_color[2], 255]
        out[~design] = [0, 0, 0, 0]
        _LOGO_CACHE[cache_key] = Image.fromarray(out, 'RGBA')
    logo_pil = _LOGO_CACHE[cache_key]
    img_pil.paste(logo_pil, (lx, ly), mask=logo_pil.split()[3])

    # --- QR (gercek) ---
    if show_qr:
        import qrcode
        qw = cfg['qr_w']
        qh = cfg['qr_h']
        qx = min(cfg['qr_cx'] - qw // 2, W_IMG - bm - qw)
        qy = row_cy - qh // 2
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M,
                           box_size=10, border=1)
        qr.add_data(str(cfg.get('number', '12345678')))
        qr.make(fit=True)
        fill_hex  = '#%02x%02x%02x' % text_color
        back_hex  = '#%02x%02x%02x' % label_color
        qr_img = qr.make_image(fill_color=fill_hex, back_color=back_hex)
        qr_img = qr_img.convert('RGB').resize((qw, qh), Image.NEAREST)
        img_pil.paste(qr_img, (qx, qy))

    # --- Numara ---
    max_nw = int((W_IMG - 2*bm) * cfg.get('num_max_w_pct', 98) / 100)
    fn  = _fit_font_cached(number, cfg['num_h'], max_nw)
    bn  = d_tmp.textbbox((0, 0), number, font=fn)
    nw  = bn[2] - bn[0]
    nh  = bn[3] - bn[1]
    nx0 = max(bm, min(W_IMG - bm - nw, cfg['num_cx'] - nw // 2 - bn[0]))
    ny  = cfg['num_cy'] + pad - nh // 2 - bn[1]
    draw.text((nx0, ny), number, fill=text_color, font=fn)

    return img_pil
