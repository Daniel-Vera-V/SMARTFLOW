from ultralytics import YOLO

# Cargar modelo base YOLOv8 nano
model = YOLO("yolov8n.pt")  # puedes usar yolov8s.pt si tienes mejor GPU

# Entrenamiento
model.train(
    data=r"C:\Users\danie\OneDrive\Escritorio\SMARTFLOW\Parking cars.v1i.yolov8\data.yaml",       # archivo YAML del dataset
    epochs=10,              # ajustable
    imgsz=640,              # tamaño de imagen
    batch=8,                # ajusta según tu RAM/GPU
    project="runs",
    name="car_detector_v1"
)
