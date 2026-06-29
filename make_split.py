import shutil, random
from pathlib import Path

random.seed(42)

DATASET = Path('dataset')
for split in ['train', 'val']:
    (DATASET / split / 'images').mkdir(parents=True, exist_ok=True)
    (DATASET / split / 'labels').mkdir(parents=True, exist_ok=True)

# --- SENTETIK: hepsi train ---
syn_imgs = sorted(Path('scenes/images').glob('*.jpg'))
for img in syn_imgs:
    lbl = Path('scenes/labels') / (img.stem + '.txt')
    shutil.copy(img, DATASET / 'train' / 'images' / img.name)
    if lbl.exists():
        shutil.copy(lbl, DATASET / 'train' / 'labels' / lbl.name)

print(f'Sentetik train: {len(syn_imgs)}')

# --- GERCEK: 150 val, kalan train ---
real_imgs = sorted(Path('real_aug2/images').glob('*.jpg'))
random.shuffle(real_imgs)
val_imgs   = real_imgs[:150]
train_imgs = real_imgs[150:]

for img in train_imgs:
    lbl = Path('real_aug2/labels') / (img.stem + '.txt')
    shutil.copy(img, DATASET / 'train' / 'images' / img.name)
    if lbl.exists():
        shutil.copy(lbl, DATASET / 'train' / 'labels' / lbl.name)

for img in val_imgs:
    lbl = Path('real_aug2/labels') / (img.stem + '.txt')
    shutil.copy(img, DATASET / 'val' / 'images' / img.name)
    if lbl.exists():
        shutil.copy(lbl, DATASET / 'val' / 'labels' / lbl.name)

print(f'Gercek train: {len(train_imgs)}  val: {len(val_imgs)}')

train_total = len(list((DATASET / 'train' / 'images').glob('*.jpg')))
val_total   = len(list((DATASET / 'val'   / 'images').glob('*.jpg')))
print(f'\nDataset hazir:')
print(f'  train: {train_total}')
print(f'  val:   {val_total}')
