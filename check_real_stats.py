from pathlib import Path
from collections import Counter

LBL_DIR = Path('real_aug/labels')

class_counts = Counter()
imgs_by_n_classes = Counter()
missing_etiket = []
low_class = []

for lbl in sorted(LBL_DIR.glob('*.txt')):
    lines = [l for l in lbl.read_text().strip().split('\n') if l]
    classes = set(int(l.split()[0]) for l in lines)
    class_counts.update(classes)
    imgs_by_n_classes[len(classes)] += 1
    if 0 not in classes:
        missing_etiket.append(lbl.name)
    if len(classes) < 8:
        low_class.append((lbl.name, len(classes), sorted(classes)))

print(f'Toplam label dosyasi: {len(list(LBL_DIR.glob("*.txt")))}')
print(f'\nSinif bazi goruntu sayisi:')
for k in sorted(class_counts):
    from gen_aug_scenes import CC
    break
for cid in range(14):
    print(f'  Sinif {cid:2d}: {class_counts[cid]:4d} goruntu')

print(f'\nSinif sayisina gore dagilim:')
for n in sorted(imgs_by_n_classes):
    print(f'  {n:2d} sinif: {imgs_by_n_classes[n]:4d} goruntu')

print(f'\nEtiket sinifi (0) eksik: {len(missing_etiket)} goruntu')

print(f'\n8den az sinif: {len(low_class)} goruntu')
for name, n, cls in low_class[:10]:
    print(f'  {name[:50]}  -> {n} sinif, classes={cls}')
