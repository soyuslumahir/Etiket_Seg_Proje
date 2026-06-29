import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt"

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

print("Model yukleniyor...")
model = YOLO(str(MODEL_PATH))
print("Hazir.")


def digits_per_tag(detections, masks_data, H, W):
    # Etiket maskelerini dilate ederek al
    etiket_items = []
    for (cid, conf, box), mask in zip(detections, masks_data):
        if cid != 0:
            continue
        m = mask.cpu().numpy()
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        binary = (m > 0.5).astype(np.uint8)
        bw = int(box[2] - box[0])
        bh = int(box[3] - box[1])
        k = max(5, int(min(bw, bh) / 8))
        binary = cv2.dilate(binary, np.ones((k, k), np.uint8))
        etiket_items.append((box, binary))

    if not etiket_items:
        return []

    etiket_boxes   = [item[0] for item in etiket_items]
    etiket_masks   = [item[1] for item in etiket_items]
    etiket_centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in etiket_boxes]

    rakamlar = [((b[0]+b[2])/2, (b[1]+b[3])/2, cid-4)
                for cid, conf, b in detections if 4 <= cid <= 13]
    header_pts_all = [((b[0]+b[2])/2, (b[1]+b[3])/2)
                      for cid, conf, b in detections if cid in (1, 2, 3)]

    def nearest_etiket(px, py):
        px_i = int(min(max(px, 0), W-1))
        py_i = int(min(max(py, 0), H-1))
        icinde = [i for i, m in enumerate(etiket_masks) if m[py_i, px_i] > 0]
        if len(icinde) == 1:
            return icinde[0]
        candidates = icinde if icinde else range(len(etiket_centers))
        return min(candidates, key=lambda i: (px-etiket_centers[i][0])**2 + (py-etiket_centers[i][1])**2)

    gruplar        = {i: [] for i in range(len(etiket_items))}
    header_gruplar = {i: [] for i in range(len(etiket_items))}

    for rx, ry, d in rakamlar:
        gruplar[nearest_etiket(rx, ry)].append((rx, ry, d))

    for hx, hy in header_pts_all:
        px_i = int(min(max(hx, 0), W-1))
        py_i = int(min(max(hy, 0), H-1))
        icinde = [i for i, m in enumerate(etiket_masks) if m[py_i, px_i] > 0]
        if icinde:
            idx = min(icinde, key=lambda i: (hx-etiket_centers[i][0])**2 + (hy-etiket_centers[i][1])**2)
            header_gruplar[idx].append((hx, hy))

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

        if header_gruplar[i]:
            hx = sum(p[0] for p in header_gruplar[i]) / len(header_gruplar[i])
            hy = sum(p[1] for p in header_gruplar[i]) / len(header_gruplar[i])
            if float(np.dot(np.array([hx, hy]) - mean, axis)) > 0:
                sorted_digits = sorted_digits[::-1]

        sonuclar.append("".join(str(d) for _, d in sorted_digits))

    return sonuclar


def run_model(img_rgb, conf_thresh=0.25):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    result = model(img_bgr, conf=conf_thresh, imgsz=2560, verbose=False)[0]

    out     = img_bgr.copy()
    overlay = out.copy()
    detections     = []
    filtered_masks = []
    filtered_cls   = []
    filtered_conf  = []

    min_etiket_area = W * H * 0.003

    if result.masks is not None:
        for mask, cls, conf, box in zip(result.masks.data, result.boxes.cls,
                                        result.boxes.conf, result.boxes.xyxy):
            cid = int(cls)
            b   = box.tolist()
            if cid == 0:
                bw     = b[2] - b[0]
                bh     = b[3] - b[1]
                area   = bw * bh
                aspect = bw / bh if bh > 0 else 0
                if area < min_etiket_area or float(conf) < 0.4 or not (0.3 <= aspect <= 3.0):
                    continue
            color = CC.get(cid, (200, 200, 200))
            m     = mask.cpu().numpy()
            m     = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0.5] = color
            detections.append((cid, float(conf), b))
            filtered_masks.append(mask)
            filtered_cls.append(cid)
            filtered_conf.append(float(conf))

        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        for mask, cls, conf, box in zip(result.masks.data, result.boxes.cls,
                                        result.boxes.conf, result.boxes.xyxy):
            cid = int(cls)
            b   = box.tolist()
            if cid == 0:
                bw     = b[2] - b[0]
                bh     = b[3] - b[1]
                area   = bw * bh
                aspect = bw / bh if bh > 0 else 0
                if area < min_etiket_area or float(conf) < 0.4 or not (0.3 <= aspect <= 3.0):
                    continue
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

    if detections:
        import torch
        masks_tensor = torch.stack(filtered_masks) if filtered_masks else None
        numaralar = digits_per_tag(detections, masks_tensor, H, W) if masks_tensor is not None else []
        counts = {}
        for cid, conf, _ in detections:
            counts[NAMES.get(cid, str(cid))] = counts.get(NAMES.get(cid, str(cid)), 0) + 1
        n_etiket = sum(1 for cid, _, _ in detections if cid == 0)
        lines = ["=== TESPIT SONUCLARI ===", f"Toplam etiket: {n_etiket}", ""]
        for i, numara in enumerate(numaralar, 1):
            lines.append(f"  Etiket {i}: {numara if numara else '(rakam yok)'}")
        lines.append("")
        for name, cnt in sorted(counts.items()):
            lines.append(f"  {name}: {cnt}")
        summary = "\n".join(lines)
    else:
        summary = "Hicbir nesne tespit edilmedi."

    return out_rgb, summary


def process_photo(image, conf):
    if image is None:
        return None, "Fotograf yuklenmedi."
    return run_model(image, conf)


def process_webcam(image, conf):
    if image is None:
        return None, ""
    return run_model(image, conf)


with gr.Blocks(title="Etiket Test - v5") as demo:
    gr.Markdown("## Etiket Tespit Sistemi — v5 (14 sinif)\n`etiket · TR · logo · QR · rakam_0–9`")

    conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence esigi")

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
