import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "runs/segment/etiket_seg_only/weights/best.pt"

print("Model yukleniyor...")
model = YOLO(str(MODEL_PATH))
print("Hazir.")


def run_model(img_rgb, conf_thresh=0.25):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    result = model(img_bgr, conf=conf_thresh, imgsz=1920, verbose=False)[0]

    out     = img_bgr.copy()
    overlay = out.copy()
    etiketler = []

    min_area = W * H * 0.003

    if result.masks is not None:
        for mask, cls, conf, box in zip(result.masks.data, result.boxes.cls,
                                        result.boxes.conf, result.boxes.xyxy):
            if int(cls) != 0:
                continue
            b  = box.tolist()
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            area   = bw * bh
            aspect = bw / bh if bh > 0 else 0
            if area < min_area or not (0.3 <= aspect <= 3.0):
                continue
            m  = mask.cpu().numpy()
            m  = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0.5] = (200, 200, 200)
            cnts, _ = cv2.findContours((m > 0.5).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 255, 0), 3)
            etiketler.append((float(conf), bw, bh, b))

        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

        for i, (conf, bw, bh, b) in enumerate(etiketler, 1):
            cx = int((b[0] + b[2]) / 2)
            cy = int((b[1] + b[3]) / 2)
            cv2.putText(out, f"#{i} {conf:.2f}", (cx-30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    lines = [f"Toplam etiket: {len(etiketler)}", ""]
    for i, (conf, bw, bh, b) in enumerate(etiketler, 1):
        aspect = bw / bh if bh > 0 else 0
        lines.append(f"  #{i}  conf={conf:.3f}  boyut={int(bw)}x{int(bh)}  aspect={aspect:.2f}")
    summary = "\n".join(lines)

    return out_rgb, summary


def process(image, conf):
    if image is None:
        return None, ""
    return run_model(image, conf)


with gr.Blocks(title="Etiket Mask Viewer") as demo:
    gr.Markdown("## Etiket Mask Viewer — sadece class 0")
    conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence esigi")
    with gr.Row():
        inp     = gr.Image(label="Fotograf", type="numpy")
        out_img = gr.Image(label="Etiket Maskeleri")
    out_text = gr.Textbox(label="Detay", lines=20)
    btn = gr.Button("Goster", variant="primary")
    btn.click(process, inputs=[inp, conf_slider], outputs=[out_img, out_text])

demo.launch(inbrowser=True)
