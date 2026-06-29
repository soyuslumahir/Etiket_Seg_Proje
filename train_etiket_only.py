"""
Sadece class 0 (etiket) icin segmentasyon modeli egitimi.
- Gercek veri: real_aug/images
- Label'lar: real_etiket_only/labels (class 0 satirlari filtrele)
- Image'lar: real_etiket_only/images altina hardlink
"""
from pathlib import Path
import shutil

if __name__ == '__main__':
    BASE     = Path(__file__).parent
    SRC_IMGS = BASE / 'real_aug' / 'images'
    SRC_LBLS = BASE / 'real_aug' / 'labels'
    DST_DIR  = BASE / 'real_etiket_only'
    DST_IMGS = DST_DIR / 'images'
    DST_LBLS = DST_DIR / 'labels'

    DST_IMGS.mkdir(parents=True, exist_ok=True)
    DST_LBLS.mkdir(parents=True, exist_ok=True)

    # 1. Filtrelenmis label dosyalari + image hardlink'leri olustur
    count = 0
    for lbl in SRC_LBLS.glob('*.txt'):
        lines = [l for l in lbl.read_text().splitlines() if l.startswith('0 ')]
        if not lines:
            continue
        for ext in ('.jpg', '.png'):
            img_src = SRC_IMGS / (lbl.stem + ext)
            if img_src.exists():
                img_dst = DST_IMGS / img_src.name
                if not img_dst.exists():
                    shutil.copy2(str(img_src), str(img_dst))
                (DST_LBLS / lbl.name).write_text('\n'.join(lines))
                count += 1
                break
    print(f'Hazir: {count} goruntu/label cifti')

    # 2. Train / val txt
    all_train = [l for l in (BASE / 'train.txt').read_text().splitlines()
                 if 'real_aug' in l]
    all_val   = (BASE / 'val.txt').read_text().splitlines()

    def remap(path_str):
        p = Path(path_str)
        dst = DST_IMGS / p.name
        return str(dst) if dst.exists() else None

    train_lines = [r for l in all_train if (r := remap(l))]
    val_lines   = [r for l in all_val   if (r := remap(l))]

    (BASE / 'train_etiket_only.txt').write_text('\n'.join(train_lines))
    (BASE / 'val_etiket_only.txt').write_text('\n'.join(val_lines))
    print(f'Train: {len(train_lines)}, Val: {len(val_lines)}')

    # 3. data.yaml
    (BASE / 'data_etiket.yaml').write_text(
        f"path: {BASE}\n"
        f"train: train_etiket_only.txt\n"
        f"val:   val_etiket_only.txt\n\n"
        f"nc: 1\n"
        f"names: ['etiket']\n"
    )
    print('data_etiket.yaml yazildi')

    # 4. Egitim
    from ultralytics import YOLO
    model = YOLO('yolov8n-seg.pt')
    model.train(
        data=str(BASE / 'data_etiket.yaml'),
        epochs=100,
        imgsz=640,
        batch=16,
        name='etiket_seg_only',
        project=str(BASE / 'runs/segment'),
        patience=20,
        lr0=0.01,
    )
