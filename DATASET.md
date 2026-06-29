# Veri Seti

Model iki farklı veri kaynağıyla eğitilmiştir: gerçek ve sentetik.

## Gerçek Veri

Gerçekten çekilmiş kulak etiketi fotoğrafları [Roboflow](https://roboflow.com) platformunda elle işaretlenerek hazırlandı. Her görüntü için etiket, TR, logo, QR ve rakam bölgeleri polygon segmentasyon maskesiyle etiketlendi. Hazırlanan dataset YOLO formatında dışa aktarıldı.

## Sentetik Veri

Gerçek verinin yetersiz kaldığı durumlarda sentetik görüntüler üretildi. Her etiket tipi için geometri ve içerik bilgisi JSON dosyalarında tanımlandı (rakamların konumu, boyutu, yazı tipi vb.). Bu JSON'lar kullanılarak programatik olarak binlerce farklı görüntü ve etiket dosyası otomatik üretildi.

## Eğitim Süreci

1. Gerçek ve sentetik görüntüler birleştirildi
2. Train / val olarak ayrıldı
3. `train_v5.py` ile YOLOv8n-seg modeli eğitildi
