from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')

    model.train(
        data='dataset/data.yaml',
        epochs=200,
        imgsz=640,
        batch=-1,
        workers=4,
        device=0,
        project='runs/seg',
        name='v5',
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        overlap_mask=False,
        save=True,
        save_period=25,
        plots=True,
        val=True,
    )
