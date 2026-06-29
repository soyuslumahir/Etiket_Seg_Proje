import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from pathlib import Path

ETIKET_MODEL_PATH = Path(__file__).parent / "runs/segment/etiket_seg_only/weights/best.pt"
DIGIT_MODEL_PATH  = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"

CC = {
    0:(200,200,200), 1:(0,200,0), 2:(0,100,255), 3:(255,165,0),
    4:(255,50,50), 5:(255,150,50), 6:(255,230,50), 7:(50,255,50),
    8:(50,255,200), 9:(50,200,255), 10:(50,50,255), 11:(200,50,255),
    12:(255,50,200), 13:(180,180,100)
}
NAMES = {
    0:'etiket', 1:'TR', 2:'logo', 3:'QR',
    4:'rakam_0', 5:'rakam_1', 6:'rakam_2', 7:'rakam_3',
    8:'rakam_4', 9:'rakam_5', 10:'rakam_6', 11:'rakam_7',
    12:'rakam_8', 13:'rakam_9'
}

# Buyuk/orta ayrimi: etiket alani goruntu alaninin %5'inden fazlaysa BUYUK
TIP_ALAN_ESIGI = 0.05

print("Modeller yukleniyor...")
etiket_model = YOLO(str(ETIKET_MODEL_PATH))
digit_model  = YOLO(str(DIGIT_MODEL_PATH))
print("Hazir.")


def digits_per_tag(etiket_items, other_detections, H, W):
    """
    etiket_items     : [(box, filled_mask_np), ...] — etiket_seg_only modelinden
    other_detections : [(cid, conf, box), ...]      — v4 modelinden (cid 1-13)
    """
    if not etiket_items:
        return []

    etiket_boxes   = [item[0] for item in etiket_items]
    etiket_masks   = [item[1] for item in etiket_items]
    etiket_centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in etiket_boxes]

    rakamlar = [((b[0]+b[2])/2, (b[1]+b[3])/2, cid-4)
                for cid, conf, b in other_detections if 4 <= cid <= 13]
    # Yon icin sadece TR (cid==1) — TR her zaman etiketin sol tarafinda
    tr_pts_all = [((b[0]+b[2])/2, (b[1]+b[3])/2)
                  for cid, conf, b in other_detections if cid == 1]

    def nearest_etiket(px, py):
        px_i = int(min(max(px, 0), W-1))
        py_i = int(min(max(py, 0), H-1))
        icinde = [i for i, m in enumerate(etiket_masks) if m[py_i, px_i] > 0]
        if len(icinde) == 1:
            return icinde[0]
        candidates = icinde if icinde else range(len(etiket_centers))
        return min(candidates, key=lambda i: (px-etiket_centers[i][0])**2 + (py-etiket_centers[i][1])**2)

    gruplar    = {i: [] for i in range(len(etiket_items))}
    tr_gruplar = {i: [] for i in range(len(etiket_items))}

    for rx, ry, d in rakamlar:
        gruplar[nearest_etiket(rx, ry)].append((rx, ry, d))

    for hx, hy in tr_pts_all:
        # Fallback: maske disinda kalsa bile en yakin etikete ata
        idx = nearest_etiket(hx, hy)
        tr_gruplar[idx].append((hx, hy))

    sonuclar = []
    for i in sorted(range(len(etiket_items)), key=lambda j: etiket_boxes[j][0]):
        icindekiler = gruplar[i]
        if len(icindekiler) < 2:
            sonuclar.append("".join(str(d) for _, _, d in icindekiler))
            continue

        pts  = np.array([[rx, ry] for rx, ry, _ in icindekiler], dtype=np.float32)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        axis = vt[0]

        projections   = [float(np.dot(np.array([rx, ry]) - mean, axis)) for rx, ry, _ in icindekiler]
        sorted_digits = sorted(zip(projections, [d for _, _, d in icindekiler]))

        if tr_gruplar[i]:
            # TR her zaman etiketin sol tarafinda: projeksiyon > 0 ise ters duz
            tr_x = sum(p[0] for p in tr_gruplar[i]) / len(tr_gruplar[i])
            tr_y = sum(p[1] for p in tr_gruplar[i]) / len(tr_gruplar[i])
            if float(np.dot(np.array([tr_x, tr_y]) - mean, axis)) > 0:
                sorted_digits = sorted_digits[::-1]

        sonuclar.append("".join(str(d) for _, d in sorted_digits))

    return sonuclar


def run_model(img_rgb, conf_thresh=0.25):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # --- Model 1: Etiket maskeleri ---
    min_etiket_area = W * H * 0.003
    res_etiket = etiket_model(img_bgr, conf=0.25, imgsz=1920, verbose=False)[0]

    etiket_items = []   # [(box, filled_mask), ...]
    out     = img_bgr.copy()
    overlay = out.copy()

    if res_etiket.masks is not None:
        for mask, conf, box in zip(res_etiket.masks.data, res_etiket.boxes.conf,
                                   res_etiket.boxes.xyxy):
            b      = box.tolist()
            bw     = b[2] - b[0]
            bh     = b[3] - b[1]
            area   = bw * bh
            aspect = bw / bh if bh > 0 else 0
            if area < min_etiket_area or not (0.3 <= aspect <= 3.0):
                continue
            m = mask.cpu().numpy()
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            # Doldur + dilate
            binary = (m > 0.5).astype(np.uint8)
            k = max(5, int(min(bw, bh) / 8))
            filled = cv2.dilate(binary, np.ones((k, k), np.uint8))
            etiket_items.append((b, filled))
            # Gorsel
            overlay[binary > 0] = (60, 60, 60)

    # --- Model 2: Rakam / TR / logo / QR ---
    res_digit = digit_model(img_bgr, conf=conf_thresh, imgsz=2560, verbose=False)[0]
    other_detections = []

    if res_digit.masks is not None:
        for mask, cls, conf, box in zip(res_digit.masks.data, res_digit.boxes.cls,
                                        res_digit.boxes.conf, res_digit.boxes.xyxy):
            cid = int(cls)
            if cid == 0:
                continue  # etiket artik model 1'den geliyor
            b     = box.tolist()
            color = CC.get(cid, (200, 200, 200))
            m     = mask.cpu().numpy()
            m     = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0.5] = color
            other_detections.append((cid, float(conf), b))

    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # --- Etiket tipi siniflamasi (cizimden once hesapla) ---
    img_area = W * H
    qr_centers = [((b[0]+b[2])/2, (b[1]+b[3])/2)
                  for cid, conf, b in other_detections if cid == 3]

    # Her QR'i en yakin etiket merkezine ata
    etiket_centers_tip = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b, _ in etiket_items]
    qr_sahipleri = set()
    for qx, qy in qr_centers:
        if etiket_centers_tip:
            idx = min(range(len(etiket_centers_tip)),
                      key=lambda i: (qx-etiket_centers_tip[i][0])**2 + (qy-etiket_centers_tip[i][1])**2)
            qr_sahipleri.add(idx)

    tipler = []
    for i, (b, filled) in enumerate(etiket_items):
        bw   = b[2] - b[0]
        bh   = b[3] - b[1]
        alan = bw * bh
        has_qr = i in qr_sahipleri
        if not has_qr:
            tipler.append("KUCUK")
        elif alan / img_area >= TIP_ALAN_ESIGI:
            tipler.append("BUYUK")
        else:
            tipler.append("ORTA")

    # Kontur + etiket tipi
    for (b, _), tip in zip(etiket_items, tipler):
        cx = int((b[0]+b[2])/2); cy = int((b[1]+b[3])/2)
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), CC[0], 2)
        cv2.putText(out, tip, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CC[0], 1, cv2.LINE_AA)

    if res_digit.masks is not None:
        for mask, cls, conf, box in zip(res_digit.masks.data, res_digit.boxes.cls,
                                        res_digit.boxes.conf, res_digit.boxes.xyxy):
            cid = int(cls)
            if cid == 0:
                continue
            b     = box.tolist()
            color = CC.get(cid, (200, 200, 200))
            m     = mask.cpu().numpy()
            m     = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)
            if cnts:
                cx = int(np.mean([c[:,0,0].mean() for c in cnts]))
                cy = int(np.mean([c[:,0,1].mean() for c in cnts]))
                cv2.putText(out, f"{NAMES.get(cid,'?')} {float(conf):.2f}",
                            (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # Sonuclar
    numaralar = digits_per_tag(etiket_items, other_detections, H, W)
    counts = {}
    for cid, conf, _ in other_detections:
        counts[NAMES.get(cid, str(cid))] = counts.get(NAMES.get(cid, str(cid)), 0) + 1

    # Tip sayilari
    tip_sayisi = {"KUCUK": tipler.count("KUCUK"), "ORTA": tipler.count("ORTA"), "BUYUK": tipler.count("BUYUK")}

    lines = ["=== TESPIT SONUCLARI ===",
             f"Toplam etiket: {len(etiket_items)}  (Kucuk:{tip_sayisi['KUCUK']}  Orta:{tip_sayisi['ORTA']}  Buyuk:{tip_sayisi['BUYUK']})",
             ""]
    sorted_indices = sorted(range(len(etiket_items)), key=lambda j: etiket_items[j][0][0])
    for sira, i in enumerate(sorted_indices):
        numara = numaralar[sira] if sira < len(numaralar) else ""
        tip    = tipler[i]
        lines.append(f"  Etiket {sira+1} [{tip}]: {numara if numara else '(rakam yok)'}")
    lines.append("")
    for name, cnt in sorted(counts.items()):
        lines.append(f"  {name}: {cnt}")
    summary = "\n".join(lines) if etiket_items or other_detections else "Hicbir nesne tespit edilmedi."

    return out_rgb, summary


def process_photo(image, conf):
    if image is None:
        return None, "Fotograf yuklenmedi."
    return run_model(image, conf)


def process_webcam(image, conf):
    if image is None:
        return None, ""
    return run_model(image, conf)


with gr.Blocks(title="Etiket Test - v6") as demo:
    gr.Markdown("## Etiket Tespit Sistemi — v6 (2 model)\n`etiket_seg_only + etiket_v4`")

    conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence esigi (rakamlar icin)")

    with gr.Tab("Fotograf"):
        with gr.Row():
            inp     = gr.Image(label="Fotograf yukle", type="numpy")
            out_img = gr.Image(label="Tespit Sonucu")
        out_text = gr.Textbox(label="Ozet", lines=20)
        btn = gr.Button("Analiz Et", variant="primary")
        btn.click(process_photo, inputs=[inp, conf_slider], outputs=[out_img, out_text])

    with gr.Tab("Canli Kamera"):
        with gr.Row():
            cam_in  = gr.Image(sources=["webcam"], streaming=True, label="Kamera", type="numpy")
            cam_out = gr.Image(label="Tespit")
        cam_text = gr.Textbox(label="Ozet", lines=15)
        cam_in.stream(process_webcam, inputs=[cam_in, conf_slider], outputs=[cam_out, cam_text])

demo.launch(inbrowser=True)
