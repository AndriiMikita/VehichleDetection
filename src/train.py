from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8x.pt')

    model.train(
        data='datasets/data.yaml',  
        epochs=100,
        imgsz=640,
        batch=16,
        patience=50,
        save=True,
        device='cpu',
        workers=8,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        cos_lr=True,
    )
    
    model.save('yolov8_tuned.pt')

if __name__ == "__main__":
    train_model()