import os
import csv
from ultralytics import YOLO
import cv2

# Ruta al modelo entrenado
model = YOLO(r"C:\Users\danie\OneDrive\Escritorio\SMARTFLOW\runs\car_detector_v14\weights\best.pt")

# Ruta de imágenes del set de test
test_images_path = "Parking cars.v1i.yolov8/test/images"

# Carpeta donde guardar predicciones
output_path = "runs/detect/test_predictions"
os.makedirs(output_path, exist_ok=True)

# CSV para guardar conteo por imagen
csv_output = "car_count_test.csv"
csv_rows = [("imagen", "autos_detectados")]

# Recorre imágenes
for filename in os.listdir(test_images_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(test_images_path, filename)
        results = model.predict(
            source=image_path,
            save=True,
            project="runs/detect",
            name="test_predictions",
            conf=0.5,
            verbose=False
        )

        # Obtener detecciones para clase "car" (asumimos clase 0)
        boxes = results[0].boxes
        detections = 0
        for box in boxes.cls:
            if int(box) == 0:
                detections += 1

        print(f"{filename}: {detections} autos detectados")
        csv_rows.append((filename, detections))

# Guardar CSV con conteos
with open(csv_output, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"\n✔ Resultados guardados en: {output_path}")
print(f"✔ CSV de conteo: {csv_output}")
