# Proje İlerlemesi

## Genel Durum
- **Başlangıç:** 18 Haziran 2026
- **Son güncelleme:** 25 Haziran 2026
- **Mevcut aşama:** İki model entegrasyonu çalışıyor, rakam sıralama düzeltildi

---

## Tamamlanan Aşamalar

### ✅ Aşama 1 — İlk Kurulum ve 5 Sınıflı Model (18 Haz)
- Roboflow'dan 54 fotoğraflık dataset indirildi
- 5 sınıf: `etiket`, `tr`, `logo`, `number`, `qr`
- YOLOv8n-det ve YOLOv8n-seg pipeline'ı kuruldu

### ✅ Aşama 2–4 — İlk Eğitimler ve OCR Denemeleri (18–19 Haz)
- YOLOv8n-det: mAP50=0.823, YOLOv8n-seg: mAP50=0.589
- EasyOCR ile rakam okuma başarılı ama güvenilir değil

### ✅ Aşama 5 — 14 Sınıfa Geçiş (22 Haz)
- `number` → `rakam_0...rakam_9` (10 ayrı sınıf)
- Model rakamları direkt sınıflandırıyor, OCR gerekmiyor

### ✅ Aşama 6–8 — Sentetik + Gerçek Veri Üretimi (22–24 Haz)
- `gen_sample.py` + `gen_2000.py`: 2000 sentetik görüntü
- `gen_real_aug.py`: 783 gerçek augmented görüntü
- 3 kritik bug düzeltildi (BUG-1, BUG-2, BUG-3 — bkz. kararlar.md)

### ✅ Aşama 9 — Val Split Düzeltmesi (24 Haz)
- **Sorun:** v3 eğitimi val=%100 sentetik ile başlatıldı → metrikler anlamsız
- **Düzeltme:** Eğitim durduruldu, val=%100 gerçek (150 foto) olarak yeniden kuruldu
- `make_splits.py`: Train=2633 (2000 synth + 633 real), Val=150 (sadece gerçek)

### ✅ Aşama 10 — etiket_v4 Eğitimi (24–25 Haz)
- 190 epoch, early stopping
- **mAP50(mask)=0.852** (gerçek val üzerinde)
- mAP50-95(mask)=0.532
- Model: `runs/segment/runs/segment/runs/segment/etiket_v4-3/weights/best.pt`
- Backup: `etiket_v4_backup.pt`

### ✅ Aşama 11 — app_v2.py (Gradio Uygulaması) (25 Haz)
- Tek geçiş, imgsz=1920
- Per-etiket rakam gruplandırma (dilate mask)
- PCA + TR/logo/QR yön düzeltmesi ile sıralama
- Etiket filtresi: conf≥0.4, alan≥%0.3, aspect 0.3–3.0
- `app_v2_stable.py` = güvenli yedek

### ✅ Aşama 12 — Etiket Mask Sorunu Tespiti (25 Haz)
- **Sorun:** Model etiket maskesinde rakam bölgelerini boş bırakıyor
- Denenen post-processing (fillPoly, convexHull, morphological closing) → hepsi başarısız
- **Kök neden:** Model görsel olarak rakam piksellerini etiket dışı öğrenmiş
- **Karar:** Ayrı 1-sınıf etiket segmentasyon modeli eğit (bkz. K-021)

### ✅ Aşama 13 — etiket_seg_only Eğitimi (25 Haz)
- 1 sınıf (class 0 = etiket), gerçek veri: 633 train + 150 val
- 100 epoch, `yolov8n-seg.pt`'den
- **mAP50(mask)=0.995, mAP50-95=0.914** — solid mask, rakam bölgeleri dahil
- Model: `runs/segment/etiket_seg_only/weights/best.pt`

### ✅ Aşama 14 — etiket_seg_only Testi (25 Haz)
- `app_etiket_only.py` ile test edildi
- Solid mask onaylandı, rakam bölgelerinde delik kalmadı
- 11/12 etiket tespit edildi (conf=0.25 ile 12/12)

### ✅ Aşama 15 — İki Model Entegrasyonu (25 Haz)
- `app_v2.py`'de iki model entegre edildi:
  - **etiket_seg_only** → etiket maskeleri (imgsz=1920)
  - **etiket_v4** → rakam/TR/logo/QR (imgsz=2560)
- Etiket maskesi koyu gri (60,60,60) ile gösteriliyor
- PCA + TR yön düzeltmesi ile rakam sıralama

### ✅ Aşama 16 — Rakam Sıralama Düzeltmesi (25 Haz)
- **Sorun:** Bazı etiketlerde rakamlar ters sıralanıyordu (87654321 vs 12345678)
- **Kök neden:** TR+logo+QR birleşik centroid kullanıyordu → ortalama merkeze yakın → yön sinyali zayıf
- **Düzeltme:** Sadece TR kullanılıyor; maske dışında kalsa bile en yakın etikete atanıyor
- **Durum:** ✅ Çalışıyor

### ✅ Aşama 17 — Etiket Tipi Sınıflaması (25 Haz)
- Her etiket KUCUK / ORTA / BUYUK olarak sınıflandırılıyor
- QR yok → KUCUK, QR var + alan < %5 → ORTA, QR var + alan ≥ %5 → BUYUK
- QR atama: en yakın etiket merkezi kullanılıyor (bbox/maske değil — komşu etiket karışmasını önler)
- Sonuç özette tip sayısı gösteriliyor, görüntüde her etiketin üzerinde tip yazıyor

### ✅ Proje Klasörü Temizliği (25 Haz)
- Eski scriptler, debug görüntüler, eski modeller ve dataset klasörleri silindi

---

### ✅ Aşama 18 — Render Sistemi Yeniden Yazıldı (26 Haz)
- **etiket_editor.py:** Gerçek logo (IMG_3824.PNG) ve gerçek QR kodu entegrasyonu
- **Pin şekli:** Dikdörtgen shaft + ayrı daire → smooth arc polygon (tek kapalı poligon)
- **Font fitting:** Sadece yükseklik kısıtı → yükseklik + genişlik kısıtı (`max_w` parametresi)
- **TR/logo/QR hizalama:** Üç elementin `cy` ortalamasıyla `row_cy` hesabı, tümü aynı yatay çizgide
- **Taşma koruması:** QR ve numara konumları body sınırları içinde kırpılıyor
- **gen_sample.py senkronizasyonu:** Pin şekli, row_cy, font fitting, numara y-pozisyonu (`y0p = ny + bn[1]`) düzeltildi
- **editor_gui.py:** Tkinter desktop editör — slider'larla canlı config düzenleme, 7h/8h butonları, dosya aç/kaydet
- **gen_multi_preview.py:** Maske PNG olarak kaydediliyor (JPEG yerine), grid yerleşimi

### ✅ Aşama 19 — Yeni Eğitim Stratejisi Kararlaştırıldı (26 Haz)
- Eski model yedeklendi: `best_v4-3_backup.pt`
- Yeni plan: 4000×3000 çok-etiketli sahneler (5–16 etiket/kare)
- `overlap_mask=False` ile eğitim (etiket maskesinde rakam bölgesi delik açılmasın)
- `mosaic=0.0` (sahneler zaten çok etiketli, mozaiğe gerek yok)
- `yolov8n-seg.yaml` ile sıfırdan eğitim (ne eski model ne COCO pretrained)
- 10 rakam sınıfı korunuyor — OCR'a gerek yok

### ✅ Aşama 20 — Çok-Etiketli Sahne Üretim Pipeline'ı (26 Haz)

#### Tespit Edilen Sorunlar
- **Pin kısmı yanlış bulunuyor:** Model, etiketin pin (iğne) bölgesini gerçek fotoğraflarda doğru segmente edemiyor. Gerçek pin 3D plastik top görünümlü, sentetik veri düz daire — model bu farkı öğrenemiyor.
- **Etiketler arası büyük boşluklar:** Grid hücrelerinden bir kısmı boş kalıyordu (`n_etiket < cols×rows`).
- **Numaralar sabit:** Tüm etiketler config'deki aynı numarayı kullanıyordu, çeşitlilik yoktu.

#### Yapılan Değişiklikler — gen_scenes.py
- **Grid tabanlı yerleşim:** Etiketler rastgele değil, cols×rows grid hücrelerine yerleştiriliyor, üst üste binme yok
- **Tam grid doldurma:** `n_etiket = cols × rows` → hiçbir hücre boş kalmıyor
- **Grid boyutu:** cols=3–4, rows=2–4 → 6–16 etiket/sahne
- **Fill faktörü:** 0.88 → 0.97 (hücreyi daha fazla dolduruyor)
- **Kenar boşlukları:** Sol/sağ 500px, üst/alt 200px
- **Açılar:** Discrete [0, 45, 90, 135, 180, 225, 270, 315]°
- **Rastgele numara:** Her etiket için bağımsız 8 haneli rastgele numara (QR ile senkron)
- **Pin şekli randomizasyonu:** %50 ihtimalle pin orijinal daire, %50 ihtimalle 12 farklı bozulmuş şekilden biri (zigzag, ellipse, polygon, blob, star, teardrop, diamond, squish, kidney, spike, wide, circle) — renk değişmez, sadece şekil → model renk/kontrast farkını öğrenir
- **Shaft bağlantısı:** Her bozulmuş şekil sonrası shaft genişliğinde dikdörtgen köprü, kopukluk önleniyor
- **Mask formatı:** PNG (JPEG bozulması olmadan)

#### Denenen ve İptal Edilen Yaklaşımlar
- **Pin 3D efekti** (radial gradient + specular) → beğenilmedi
- **Pin renk değişimi** (beyaz/siyah/gri) → beğenilmedi
- **Pin perspektif warp** (tüm üst bölge) → kötü görünüm
- **Pin rotasyon warp** (shoulder_y'den eğme) → kötü görünüm

#### Yeni Dosyalar
- **gen_scenes.py** ← 4000×3000 çok-etiketli sahne üretici (ana pipeline)
- **preview_scenes.py** ← mask overlay önizleme (scenes/previews/)
- **scenes/images/** ← üretilen sahne görselleri (.jpg)
- **scenes/masks/** ← mask görüntüleri (.png)
- **scenes/labels/** ← YOLO format annotation (.txt)
- **scenes/previews/** ← mask overlay önizlemeleri

#### Eğitim Planı (Sonraki Adım)
- **Train:** 500 sentetik sahne + 633 gerçek görsel
- **Val:** 150 gerçek görsel (%100 gerçek)
- `overlap_mask=False`, `mosaic=0.0`, sıfırdan eğitim

---

## Aktif Aşama

500 sentetik sahne üretimi sürüyor (arka planda).
**Sıradaki:** Sahne üretimi bitince splits oluştur → eğitimi başlat.

---

## Dosya Yapısı (Güncel)

```
C:\Users\soyus\Desktop\etiket-proje\
├── runs/segment/
│   ├── runs/segment/runs/segment/etiket_v4-3/weights/best.pt   ← 14-sınıf rakam modeli
│   └── etiket_seg_only/weights/best.pt                         ← 1-sınıf etiket modeli
├── best_v4-3_backup.pt        ← v4-3 güvenli yedek (26 Haz)
├── app_v2.py                  ← Ana uygulama (2 model)
├── app_v2_stable.py           ← Güvenli yedek (tek model, imgsz=2560)
├── etiket_editor.py           ← Tek etiket render (gerçek logo/QR, smooth pin)
├── gen_sample.py              ← (visual, mask) çifti üretimi
├── gen_scenes.py              ← 4000×3000 çok-etiketli sahne üretici
├── preview_scenes.py          ← Mask overlay önizleme scripti
├── gen_multi_preview.py       ← 12 etiketli preview grid
├── editor_gui.py              ← Tkinter config editörü
├── configs/                   ← 1000 JSON config (500 tip1, 500 tip2)
├── scenes/                    ← Üretilen sahneler (images/ masks/ labels/ previews/)
├── dataset_aug/               ← Eski sentetik veriler (633 gerçek augmented)
├── IMG_3824.PNG               ← Tarım Bakanlığı logosu
└── data.yaml                  ← 14 sınıf
```

---

## Metrik Özeti

| Model | Val | mAP50(mask) | mAP50-95(mask) |
|---|---|---|---|
| etiket_v1 | %100 sentetik | ~0.890 | — (geçersiz) |
| etiket_v4 | %100 gerçek (150) | **0.852** | 0.532 |
| etiket_seg_only | %100 gerçek (150) | **0.995** | 0.914 |
