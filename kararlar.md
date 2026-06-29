# Proje Kararları ve Teknik Kurallar

Projedeki önemli teknik ve tasarım kararlarının kaydı.
**YENİ AGENT:** Bu dosyadaki kurallar çok sancılı debug süreçlerinden çıktı. Değiştirme.

---

## KRITIK BUG DÜZELTMELERİ (GERİ ALMA)

### BUG-1 — cv2.fillPoly Zorunluluğu (gen_sample.py)
**Dosya:** `gen_sample.py` — render_pair() fonksiyonu içinde etiket maske çizimi
**Sorun:** PIL'in `ImageDraw.polygon()` non-convex şekillerde scan-line fill hatası yapıyor → etiket gövdesinde adacıklar bırakıyor → YOLO polygon bozuluyor
**Düzeltme:**
```python
# DOĞRU — cv2.fillPoly kullan
m=np.zeros((H,W_IMG,3),dtype=np.uint8)
pts_cv=np.array(pts,dtype=np.int32)
cv2.fillPoly(m,[pts_cv],(200,200,200))
cv2.rectangle(m,(cx-sh,pcy-pr),(cx+sh,sb+10),(200,200,200),-1)
cv2.ellipse(m,(cx,pcy),(pr,pr),0,0,360,(200,200,200),-1)
mask=Image.fromarray(m)
# YANLIŞ — draw.polygon(pts, fill=(200,200,200)) KULLANMA
```
**Not:** `cv2.rectangle` ile shaft bölgesi pcy-pr'den sb+10'a kadar uzatıldı → pin-body birleşimi için şart.

---

### BUG-2 — Non-Black Contour (gen_2000.py mask_to_yolo)
**Dosya:** `gen_2000.py` — mask_to_yolo() fonksiyonu, class 0 (etiket) işleme
**Sorun:** Rakam renkleri (DIGIT_COLORS) etiket piksellerini üstüne yazıyor → class 0 için `(200,200,200)` rengini arayınca 10 ayrık ada buluyor → 10 contour → YOLO çıktısı bozuluyor
**Düzeltme:** class 0 için renk aramak yerine tüm non-black piksellerin dış konturunu al:
```python
def mask_to_yolo(msk_rgb, W, H):
    lines = []
    for cid, color in CC.items():
        if cid == 0:
            # Etiket: tüm non-black piksellerin dış konturunu al
            binary = (msk_rgb.max(axis=2) > 10).astype(np.uint8) * 255
            min_area = 1000
        else:
            c = np.array(color, dtype=np.int32)
            diff = np.abs(msk_rgb.astype(np.int32) - c).max(axis=2)
            binary = (diff < 5).astype(np.uint8) * 255
            min_area = 50
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            eps = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            coords = ' '.join(f'{x/W:.6f} {y/H:.6f}' for x, y in approx)
            lines.append(f'{cid} {coords}')
    return lines
```
**Neden çalışıyor:** Etiket her zaman resmin en büyük nesnesi. Tüm class renkleri siyah arka plan üzerinde → non-black = etiketin tamamı (pin dahil, rakamlar dahil, hepsi bir bütün).

---

### BUG-3 — Albumentations additional_targets Hatası (aug_new.py)
**Dosya:** `aug_new.py` — stage1 ve stage6 içindeki A.Compose çağrıları
**Sorun:** `additional_targets={'mask': 'image'}` kullanıldığında:
1. Perspective ve Affine dönüşümlerde boş alanlar `BG_COLOR_RGB=(209,196,38)` SARI ile dolduruluyor → maske üzerinde sahte sarı piksel bölgesi → non-black piksel dedeksiyonu bozuluyor
2. ElasticTransform INTER_LINEAR interpolasyon kullanıyor → renk karışması → sınıf kirlenmesi
**Düzeltme:**
```python
# DOĞRU
stage1 = A.Compose([...], additional_targets={'mask': 'mask'})
stage6 = A.Compose([...], additional_targets={'mask': 'mask'})

# YANLIŞ — bunları kullanma
# additional_targets={'mask': 'image'}
```
**Neden önemli:** `'mask'` tipi belirtildiğinde Albumentations:
- Doldurma için `(0,0,0)` kullanır (BG_COLOR değil)
- INTER_NEAREST interpolasyon kullanır (renk karışması olmaz)

---

## Teknik Kurallar

### K-001 — YOLOv8 Seçimi
**Tarih:** 18 Haziran 2026
**Karar:** Klasik OpenCV pipeline yerine YOLOv8 kullanılacak
**Gerekçe:** TR tespiti %75, rakam okuma %33 başarıyla başladı. Farklı açı, ışık, mesafeye dayanıklılık için deep learning gerekli.

---

### K-002 — Segmentation Tercih Edildi
**Tarih:** 18 Haziran 2026
**Karar:** Detection'dan (mAP50=0.823) iyi olmasına rağmen segmentation kullanılacak (mAP50=0.589 idi, şimdi 0.890)
**Gerekçe:** Piksel-düzeyi maske rakam okuma ve QR decode için daha kullanışlı.

---

### K-003 — 14 Sınıf (number → rakam_0...9)
**Tarih:** 22 Haziran 2026
**Karar:** `number` sınıfı kaldırıldı, her rakam ayrı sınıf oldu
**Gerekçe:** Model hangi rakamın "3" hangisinin "7" olduğunu bilmiyordu. Kimlik numarasını direkt modelden okumak için gerekli.
**Sınıflar:** etiket(0), TR(1), logo(2), QR(3), rakam_0–9(4–13)

---

### K-004 — Sentetik Veri Üretimi
**Tarih:** 22 Haziran 2026
**Karar:** 1000 JSON config → kod ile sentetik veri üretilecek
**Gerekçe:** Etiketin yapısı deterministik (sarı arka plan, belirli konumlar) → gerçek fotoğraf toplamak zaman alır.
**Sonuç:** 1000 × 2 = 2000 augmented görüntü (gen_2000.py)

---

### K-005 — Hibrit Dataset (Sentetik + Gerçek)
**Tarih:** 24 Haziran 2026 (revize)
**Karar:** 2000 sentetik + 780 gerçek augmented = ~2780 toplam
**Önemli:** Gerçek fotoğraflarda `rakam_0...rakam_9` etiketi OLMALIDIR — olmadan model gerçek fotoğraflarda rakam göremez

---

### K-006 — RGB Maske Yaklaşımı
**Tarih:** 22 Haziran 2026
**Karar:** Her sınıf için sabit unique RGB renk, tek PNG dosyasında tüm maskeler
**Detay:** `diff < 5` toleransı — JPEG artefaktlarına karşı dayanıklı
**CC dict:** (bkz. ilerleme.md → Renk Kodları bölümü)

---

### K-007 — RETR_EXTERNAL (Contour Mode)
**Tarih:** 24 Haziran 2026 (değiştirildi)
**Karar:** Tüm sınıflar için `cv2.RETR_EXTERNAL` kullanılır
**Gerekçe (değişiklik nedeni):** Logo ve QR artık basit şekil (ellipse/dikdörtgen) → iç kontur yok. Eski RETR_LIST kaldırıldı.

---

### K-008 — INTER_NEAREST Zorunluluğu
**Tarih:** 22 Haziran 2026
**Karar:** Maske augmentasyonunda her zaman `cv2.INTER_NEAREST` kullanılacak
**Gerekçe:** INTER_LINEAR kullanıldığında piksel renkleri karışır: (200,200,200) ve (0,200,0) arasındaki piksel (100,200,100) olur → hiçbir sınıfa karşılık gelmez.
**Kural:** Rotation, scale, perspective, elastic → maskede HEPSI INTER_NEAREST

---

### K-009 — Augmentation Parametreleri (aug_new.py)
**Tarih:** 24 Haziran 2026

| Parametre | Değer |
|---|---|
| Rotation | 8 discrete açı (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) |
| Perspective scale | 0.02–0.05 |
| Affine shear | ±3° |
| Scale (kamera mesafesi) | 0.60–0.90× |
| Hue shift | ±7° |
| Saturation | ±18% |
| GaussNoise | var 3–12 |
| JPEG quality | 60–95, p=0.5 |
| ElasticTransform | alpha=15, sigma=4, p=0.4 |

**Yasaklar:**
- HorizontalFlip yok (etiket gerçekte 180° ters gelmez)
- Ara açı rotasyon yok (yalnızca 8 discrete açı)
- Maske üzerine fotometrik transform yok (lighting, noise, renk sadece image'e)

---

### K-010 — Logo ve QR Basitleştirmesi
**Tarih:** 24 Haziran 2026
**Karar:** Logo → ellipse (daire), QR → dolu dikdörtgen
**Gerekçe:** Logo zaten yuvarlak, QR'da piksel maskesi gerekmez. Model kapasitesini TR ve rakam öğrenmesine ayır.

---

### K-011 — YOLOv8n-seg (Nano)
**Tarih:** 23 Haziran 2026
**Karar:** En küçük model
**Gerekçe:** ~3000 görüntü için yeterli. Büyük model overfit yapar. RTX 4070 Laptop (~4GB VRAM) ile uyumlu.

---

### K-012 — Python API (CLI Yerine)
**Tarih:** 23 Haziran 2026
**Karar:** `yolo train ...` CLI komutu yerine Python API
**Gerekçe:** `No module named ultralytics.cfg.__main__` hatası — CLI Python 3.11 ile uyumsuz
**Kullanım:**
```python
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.train(data='data.yaml', epochs=200, imgsz=640, batch=16, name='etiket_v2')
```

---

### K-013 — Sıfırdan Eğitim (Fine-tune Değil)
**Tarih:** 24 Haziran 2026
**Karar:** `best.pt`'den fine-tune değil, `yolov8n-seg.pt`'den sıfırdan
**Gerekçe:** Eski eğitimde 290 gerçek fotoğraf rakam etiketi olmadan girdi → model "gerçekte rakam yok" öğrendi (negatif öğrenme). Bu bias fine-tune ile temizlenemez.

---

### K-014 — Gerçek Veri: Sadece 5 Fotoğraf (Yeniden Etiketlenmiş)
**Tarih:** 24 Haziran 2026
**Karar:** Eski 290 fotoğraf çıkarıldı. Yeni: 5 ham fotoğraf + crop augmentation
**Neden 5:** Bu 5 fotoğraf Roboflow'da `rakam_0...rakam_9` ile doğru etiketlendi.
**Sonuç:** 5 fotoğraf × crop → 39 kaynak × 20 aug = 780 görüntü

---

### K-015 — gen_real_aug.py Label Dönüşümü
**Tarih:** 24 Haziran 2026
**Karar:** Geometric augmentation (flip, rotation) sırasında YOLO polygon koordinatları matematiksel olarak dönüştürülüyor
**Yöntem:**
- Vertical flip: `y_new = H-1-y`
- Rotation angle θ: `xr = cos(θ)*xc - sin(θ)*yc + W/2`, `yr = sin(θ)*xc + cos(θ)*yc + H/2`
- Crop: parent label koordinatlarından crop bbox çıkarılıyor

---

### K-016 — Multi-Contour Sıfır Testi
**Tarih:** 24 Haziran 2026
**Tanım:** Üretilen her etiket görüntüsünün label dosyasında class 0 (etiket) için maksimum 1 contour olmalı
**Test kodu:**
```python
import random
from pathlib import Path

labels = list(Path('dataset_aug/labels').glob('*.txt'))
multi = 0
for p in random.sample(labels, min(50, len(labels))):
    lines = p.read_text().strip().split('\n')
    etiket_count = sum(1 for l in lines if l.startswith('0 '))
    if etiket_count > 1:
        multi += 1
        print(f'MULTI: {p.name} — {etiket_count} contour')
print(f'Multi-contour: {multi}/{min(50, len(labels))}')
```
**Beklenen:** 0/50 (tüm düzeltmeler uygulandıktan sonra test edildi ve %0 çıktı)

---

### K-017 — OneDrive Tuzağı
**Tarih:** 24 Haziran 2026
**Sorun:** `C:\Users\soyus\OneDrive\Desktop\etiket-proje` — OneDrive sync durduğunda dosyalar "çevrimiçi" moda geçiyor → `open()` → FileNotFoundError
**Çözüm:** Her zaman `C:\Users\soyus\Desktop\etiket-proje` kullan
**gen_2000.py çalıştırma:**
```powershell
cd C:\Users\soyus\Desktop\etiket-proje
python gen_2000.py
```
NOT: `cd` ile dizin değişimi şart çünkü `CONFIGS_DIR = Path('configs')` relative path.

---

### K-018 — Val=%100 Gerçek Fotoğraf
**Tarih:** 24 Haziran 2026
**Karar:** Val setine kesinlikle sentetik veri girmiyor
**Gerekçe:** Sentetik val üzerindeki metrikler (mAP50=0.992) yanıltıcıydı — model sentetik veriyi ezber yapıyor, gerçekte çalışmıyordu.
**Kural:** Val = sadece gerçek fotoğraf. Train'e sentetik eklenebilir.

---

### K-019 — imgsz=1920 (Inference)
**Tarih:** 25 Haziran 2026
**Karar:** Inference'ta imgsz=1920 kullanılıyor (640 değil)
**Gerekçe:** 4000×3000 fotoğraflarda etiketler ~80px olduğundan 640px'de tespit edilemiyor. 1920px en iyi hız/doğruluk dengesi.
**Not:** Eğitim hâlâ imgsz=640 — inference'ta ölçekleme YOLO tarafından yönetiliyor.

---

### K-020 — app_v2_stable.py Güvenli Nokta
**Tarih:** 25 Haziran 2026
**Karar:** app_v2_stable.py dokunulmadan saklanır, her büyük değişiklikten önce güncellenir
**Gerekçe:** Bozulan versiyonu geri almak için referans nokta.

---

### K-021 — İki Model Mimarisi (Etiket + Rakam Ayrımı)
**Tarih:** 25 Haziran 2026
**Karar:** Etiket segmentasyonu ve rakam tespiti ayrı modellerle yapılacak
**Gerekçe:** 14-sınıf model etiket maskesini rakam bölgelerinde delikli üretiyor. Post-processing (fillPoly, convexHull, morphological closing) çözüm vermiyor. 1-sınıf odaklı model mAP50(mask)=0.995 başardı.
**Mimari:**
- `etiket_seg_only` → etiket sınırları (solid mask)
- `etiket_v4` → rakam/TR/logo/QR konumları
**Hız:** RTX 4070'te iki inference ~300-400ms, kabul edilebilir.

---

### K-022 — Adım Adım İlerleme Kuralı
**Tarih:** 25 Haziran 2026
**Karar:** Her yeni component önce tek başına test edilir, onay sonrası entegre edilir
**Kural:** Eğitim bitti → test et → onay → birleştir. Onay olmadan bir sonraki adıma geçilmez.

---

### K-023 — Rakam Sıralama: Sadece TR ile Yön Düzeltmesi
**Tarih:** 25 Haziran 2026
**Karar:** PCA ekseni boyunca rakam sıralamada yön düzeltmesi için **sadece TR (cid==1)** kullanılır. Logo (cid==2) ve QR (cid==3) yön hesabına dahil edilmez.
**Gerekçe:** TR etikette her zaman sol tarafta, logo/QR sağ tarafta. Hepsinin centroidini alınca orta çıkıyor, yön sinyali belirsizleşiyor. TR tek başına güvenilir bir "sol marker" görevi görüyor.
**Ek:** TR maske dışında kalırsa nearest-etiket fallback ile yine de ilgili etikete atanıyor.
**Mantık:** TR projeksiyon > 0 → TR sağda (ters) → diziyi ters çevir. TR projeksiyon ≤ 0 → TR solda (doğru) → değiştirme.
**imgsz notu:** etiket_seg_only=1920, digit_model=2560 (rakamlar küçük olduğundan daha yüksek).

---

### K-024 — Etiket Tipi Sınıflaması
**Tarih:** 25 Haziran 2026
**Karar:** Her etiket KUCUK / ORTA / BUYUK olarak sınıflandırılır.
**Mantık:**
- QR tespit edilmedi → KUCUK (sadece TR + logo)
- QR var + etiket alanı < görüntü alanının %5'i → ORTA
- QR var + etiket alanı ≥ görüntü alanının %5'i → BUYUK
**QR atama:** Her QR en yakın etiket merkezine atanır (bbox veya maske kullanılmaz). Komşu etiketin QR'ının yanlış etikete atanmasını önler.
**Eşik gerekçesi:** Test görüntülerinde küçük/orta etiketler %2.5–3.5, büyük etiketler %7.5–12.6 alan kaplıyor. %5 eşiği bu iki grubu temiz ayırıyor.

---

### K-025 — overlap_mask=False Zorunluluğu
**Tarih:** 26 Haziran 2026
**Karar:** Tüm eğitimlerde `overlap_mask=False` kullanılacak
**Gerekçe:** overlap_mask=True (varsayılan) ile rakam maskeleri etiket maskesinin piksellerini eziyor. Model "rakam bölgesi = etiket değil" öğreniyor → inference'ta etiket maskesinde rakam bölgelerinde delik açılıyor. Bu sorun iki model mimarisine geçişi zorunlu kılmıştı (K-021). overlap_mask=False ile etiket ve rakam maskeleri aynı bölgede birlikte var olabiliyor.
**Kullanım:** `model.train(..., overlap_mask=False)`

---

### K-026 — Çok Etiketli Sentetik Sahne Stratejisi
**Tarih:** 26 Haziran 2026
**Karar:** Eğitim verisi tek etiketli görüntüler değil, 4000×3000 sahne başına 5–16 etiket içeren görüntülerden oluşacak
**Gerekçe:** Tek etiketli eğitim verisi + YOLOv8 mozaik augmentasyonu ile çok etiketli sahneler simüle ediliyordu. Mozaik kontrolsüz ve gerçek dışı geçişler yaratıyordu. Çok etiketli sahnelerin elle üretilmesi daha kontrollü ve gerçeğe yakın.
**Parametreler:** mosaic=0.0 (kapalı), imgsz=1280 veya tile yaklaşımı

---

### K-027 — Sıfırdan Eğitim, COCO Pretrained Yok
**Tarih:** 26 Haziran 2026
**Karar:** `yolov8n-seg.yaml` ile sıfırdan eğitim. Ne eski custom model ne COCO pretrained ağırlıkları kullanılacak.
**Gerekçe:** COCO pretrained ağırlıkları backbone'u hazır getirir ama kullanıcı bunu istemiyor — sadece sentetik etiket verisiyle öğrenmesi isteniyor. Eski model hatalı maskelerle öğrendiği için fine-tune değil sıfırdan başlamak gerekiyor.

---

### K-028 — gen_sample.py / etiket_editor.py Senkronizasyon Kuralı
**Tarih:** 26 Haziran 2026
**Kural:** etiket_editor.py'de yapılan her geometrik/pozisyon değişikliği gen_sample.py'ye de uygulanmalı. İkisi arasındaki fark mask-görsel hizalama bozukluğuna yol açar.
**Kritik noktalar:**
- row_cy hesabı (TR/logo/QR ortalaması)
- Font fitting: `_ec_fit(number, cfg['num_h'], max_nw)` — max_w parametresi şart
- Numara y pozisyonu: `y0p = ny + bn[1]` (bn[1] sıfır değilse glyph kayar)
- QR ve numara taşma koruması: `min(cfg['qr_cx'] - qw//2, W_IMG - bm - qw)`

---

## Sonuç Metrikleri (İlk Eğitim — Referans)

| Metrik | 5 sınıf (eski) | 14 sınıf (1. round) | Hedef (v2) |
|---|---|---|---|
| mAP50 (mask) | 0.589 | 0.890 | >0.85 |
| mAP50-95 (mask) | 0.438 | 0.731 | >0.70 |
| Gerçek foto rakam tespiti | — | BAŞARISIZ | BAŞARILI |
| Dataset boyutu | 54 | 3290 | ~2780 |

**NOT:** 0.890 mAP'ye rağmen gerçek fotoğraflarda rakam çıkmıyordu çünkü eğitim verisinde gerçek fotoğraflara rakam etiketi konulmamıştı. v2 bunu çözecek.
