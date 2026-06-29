import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

MODEL  = 'best.pt'
VAL_IMGS = Path('dataset/val/images')
VAL_LBLS = Path('dataset/val/labels')
CONF   = 0.25
IMGSZ  = 640

NAMES = {0:'etiket',1:'TR',2:'logo',3:'QR',
         4:'rakam_0',5:'rakam_1',6:'rakam_2',7:'rakam_3',8:'rakam_4',
         9:'rakam_5',10:'rakam_6',11:'rakam_7',12:'rakam_8',13:'rakam_9'}

model = YOLO(MODEL)

# Per-class: TP, FP, FN sayilari
tp = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)

miss_images  = []   # hic tespit yok
extra_images = []   # fazla tespit var

img_paths = sorted(VAL_IMGS.glob('*.jpg'))

for img_path in img_paths:
    lbl_path = VAL_LBLS / (img_path.stem + '.txt')
    if not lbl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Ground truth
    gt_classes = []
    for line in lbl_path.read_text().strip().split('\n'):
        if line:
            gt_classes.append(int(line.split()[0]))

    # Prediction
    result = model(img, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    pred_classes = [int(c) for c in result.boxes.cls] if result.boxes else []

    gt_count   = defaultdict(int)
    pred_count = defaultdict(int)
    for c in gt_classes:   gt_count[c]   += 1
    for c in pred_classes: pred_count[c] += 1

    all_cls = set(gt_count) | set(pred_count)
    for c in all_cls:
        g = gt_count[c]
        p = pred_count[c]
        matched = min(g, p)
        tp[c] += matched
        fp[c] += max(0, p - g)
        fn[c] += max(0, g - p)

    # Eksik/fazla tespitli gorseller
    total_gt   = len(gt_classes)
    total_pred = len(pred_classes)
    if total_pred < total_gt * 0.7:
        miss_images.append((img_path.name, total_gt, total_pred))
    elif total_pred > total_gt * 1.5:
        extra_images.append((img_path.name, total_gt, total_pred))

print('='*60)
print(f'{"Sinif":<12} {"TP":>6} {"FP":>6} {"FN":>6} {"Precision":>10} {"Recall":>8}')
print('-'*60)
for cid in range(14):
    t = tp[cid]; f = fp[cid]; n = fn[cid]
    prec = t/(t+f) if (t+f) > 0 else 0
    rec  = t/(t+n) if (t+n) > 0 else 0
    name = NAMES[cid]
    print(f'{name:<12} {t:>6} {f:>6} {n:>6} {prec:>10.2%} {rec:>8.2%}')

print('='*60)
total_tp = sum(tp.values())
total_fp = sum(fp.values())
total_fn = sum(fn.values())
print(f'{"TOPLAM":<12} {total_tp:>6} {total_fp:>6} {total_fn:>6}')

print(f'\nEksik tespit ({len(miss_images)} gorsel):')
for name, gt, pred in miss_images[:10]:
    print(f'  {name}: gt={gt} pred={pred}')

print(f'\nFazla tespit ({len(extra_images)} gorsel):')
for name, gt, pred in extra_images[:10]:
    print(f'  {name}: gt={gt} pred={pred}')
