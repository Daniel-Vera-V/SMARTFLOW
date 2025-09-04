import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO
import cv2

# === CONFIGURACIÓN ===
  # Modelo entrenado
carpeta_test = r"C:\Users\danie\OneDrive\Escritorio\SMARTFLOW\Parking cars.v1i.yolov8\test\images"               # Carpeta con imágenes test
csv_salida = "registro_autos_test.csv"                      # Archivo CSV
intervalo = 10  # segundos entre cada imagen

# === CARGA MODELO ===
model = YOLO(r"C:\Users\danie\OneDrive\Escritorio\SMARTFLOW\runs\car_detector_v14\weights\best.pt")

# === PREPARAR CSV ===
if not os.path.exists(csv_salida):
    with open(csv_salida, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "imagen", "autos_detectados"])

# === OBTENER LISTA DE IMÁGENES ===
imagenes = [img for img in os.listdir(carpeta_test) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
imagenes.sort()  # opcional: ordena por nombre

# === LOOP POR IMÁGENES ===
print(f"🚦 Iniciando detección en carpeta '{carpeta_test}'...\n")

for img in imagenes:
    ruta = os.path.join(carpeta_test, img)

    # Procesar imagen
    results = model.predict(ruta, conf=0.5, verbose=False)

    # Contar autos (clase 0 = auto)
    detecciones = 0
    for cls in results[0].boxes.cls:
        if int(cls) == 0:
            detecciones += 1

    # Guardar en CSV
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_salida, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, img, detecciones])

    print(f"[{now}] {img} → Autos: {detecciones}")

    time.sleep(intervalo)

print("\n✅ Procesamiento completado.")
