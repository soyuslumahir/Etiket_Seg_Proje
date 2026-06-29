# Hayvan Kulak Etiketi Tespit Sistemi

Hayvanlara takılan plastik kulak etiketlerini fotoğraftan otomatik olarak tespit eden ve üzerindeki 8 haneli kimlik numarasını okuyan bir bilgisayarlı görü sistemi.

YOLOv8 segmentasyon modeli kullanılarak geliştirilmiştir. Etiket, TR yazısı, logo, QR kod ve 0–9 rakamları olmak üzere 14 sınıfı ayırt edebilir. Rakamların sırasını ve yönünü otomatik olarak belirleyerek doğru numarayı üretir.

Gradio tabanlı web arayüzü üzerinden fotoğraf yükleyerek veya canlı kamera ile kullanılabilir.

## Kurulum

Python 3.9+ gereklidir.

```bash
git clone https://github.com/soyuslumahir/etiket-proje.git
cd etiket-proje
pip install -r requirements.txt
```

## Kullanım

```bash
python app.py
```

Uygulama başladığında tarayıcı otomatik olarak açılır.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `app.py` | Ana Gradio uygulaması — görüntü yükle, modeli çalıştır, numaraları oku |
| `best.pt` | Eğitilmiş YOLOv8n-seg modeli |
| `train_v5.py` | Modeli yeniden eğitmek için kullanılan script |
| `eval_val.py` | Validasyon seti üzerinde per-class precision/recall hesaplar |
| `test_raw.py` | Ham 4000×3000 fotoğraflar üzerinde toplu test yapar |
| `data.yaml` | YOLO konfigürasyonu — sınıf isimleri ve dataset yolları |
| `requirements.txt` | Gerekli Python paketleri |

## Model

`best.pt` dosyası repoya dahildir, ek bir indirme gerekmez.  
Validasyon seti üzerinde etiket/TR/logo/QR sınıfları %100, rakam sınıfları %92–99 precision değerine ulaşmıştır.
