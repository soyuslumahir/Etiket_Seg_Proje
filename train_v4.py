from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')
    model.train(
        data='data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        name='etiket_v4',
        project='runs/segment/runs/segment',
        patience=30,
    )
