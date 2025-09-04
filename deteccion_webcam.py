import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

# Cargar el modelo entrenado
model = YOLO(r"C:\Users\danie\OneDrive\Escritorio\SMARTFLOW\runs\car_detector_v14\weights\best.pt")

# Crear carpeta para guardar las imágenes
output_folder = "capturas_webcam"
os.makedirs(output_folder, exist_ok=True)

# Crear CSV y DataFrame
csv_file = "detecciones.csv"
df = pd.DataFrame(columns=["timestamp", "nombre_img", "n_autos"])

# Captura desde webcam
cap = cv2.VideoCapture(0)  # 0 es la webcam por defecto

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

for i in range(7):  # Máximo 7 imágenes
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen.")
        continue

    # Generar nombre e imagen
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    nombre_img = f"img_{i+1}_{timestamp}.jpg"
    ruta_img = os.path.join(output_folder, nombre_img)
    cv2.imwrite(ruta_img, frame)
    print(f"✅ Imagen capturada: {nombre_img}")

    # Aplicar modelo
    results = model(ruta_img)
    boxes = results[0].boxes
    n_autos = len(boxes)

    # Agregar al DataFrame
    df.loc[len(df)] = {
        "timestamp": timestamp,
        "nombre_img": nombre_img,
        "n_autos": n_autos
    }

    # Esperar 10 segundos antes de la siguiente captura
    time.sleep(10)

# Guardar CSV
df.to_csv(csv_file, index=False)
print("✅ CSV guardado:", csv_file)

# Liberar cámara
cap.release()
cv2.destroyAllWindows()
