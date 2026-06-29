import random
from pathlib import Path

random.seed(42)

BASE = Path(r"C:\Users\soyus\Desktop\etiket-proje")
SYNTH_IMG = BASE / "dataset_aug" / "images"
REAL_IMG  = BASE / "real_aug" / "images"

synth_all = sorted(SYNTH_IMG.glob("*.jpg"))
real_all  = sorted(REAL_IMG.glob("*.jpg"))

random.shuffle(synth_all)
random.shuffle(real_all)

VAL_REAL  = 150

val_real   = real_all[:VAL_REAL]
train_real = real_all[VAL_REAL:]

train_paths = synth_all + train_real
val_paths   = val_real

random.shuffle(train_paths)
random.shuffle(val_paths)

with open(BASE / "train.txt", "w") as f:
    f.write("\n".join(str(p) for p in train_paths) + "\n")

with open(BASE / "val.txt", "w") as f:
    f.write("\n".join(str(p) for p in val_paths) + "\n")

print(f"Train: {len(train_paths)}  ({len(synth_all)} synth + {len(train_real)} real)")
print(f"Val:   {len(val_paths)}  (0 synth + {len(val_real)} real)")

# Label eslesmesi dogrula
missing = 0
for p in train_paths + val_paths:
    lbl = p.parent.parent / "labels" / (p.stem + ".txt")
    if not lbl.exists():
        print(f"EKSIK LABEL: {lbl}")
        missing += 1
if missing == 0:
    print("Tum labellar mevcut. Hazir.")
else:
    print(f"HATA: {missing} eksik label!")
