# Veri Seti

Model iki farklı veri kaynağıyla eğitilmiştir: gerçek ve sentetik.

## Gerçek Veri

Gerçekten çekilmiş kulak etiketi fotoğrafları [Roboflow](https://roboflow.com) platformunda elle işaretlenerek hazırlandı. Her görüntü için etiket, TR, logo, QR ve rakam bölgeleri polygon segmentasyon maskesiyle etiketlendi. Hazırlanan dataset YOLO formatında dışa aktarıldı.

## Sentetik Veri

Gerçek verinin yetersiz kaldığı durumlarda sentetik görüntüler programatik olarak üretildi.

**JSON tabanlı geometri tanımı**  
Projede iki farklı etiket tipi (QR'li, QR'siz) ve her tipin birden fazla boyut varyantı bulunmaktadır. Her varyant için etiketin fiziksel geometrisi - rakamların konumu, boyutu, TR/logo/QR bölgelerinin yerleri - ayrı bir JSON dosyasında piksel hassasiyetinde tanımlandı. Bu sayede her üretilen görüntüde segmentasyon maskeleri otomatik olarak hesaplanabildi, elle etiketleme gerekmedi.

**Görüntü üretimi**  
JSON'lardaki geometri bilgisi kullanılarak rastgele rakam kombinasyonları, farklı arka planlar ve aydınlatma koşullarıyla binlerce görüntü üretildi. Üretilen her görüntüyle birlikte YOLO formatında etiket dosyası da otomatik oluşturuldu.

**Augmentation**  
Hem sentetik hem gerçek görüntülere perspektif bozulması, renk değişimi, bulanıklaştırma ve gürültü gibi augmentasyonlar uygulanarak modelin farklı çekim koşullarına karşı dayanıklılığı artırıldı.

## Eğitim Süreci

1. Gerçek ve sentetik görüntüler birleştirildi
2. Train / val olarak ayrıldı
3. `train_v5.py` ile YOLOv8n-seg modeli eğitildi
