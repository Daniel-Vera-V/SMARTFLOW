import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import os
import torch
 
cap = cv2.VideoCapture(r"C:\Users\danie\OneDrive\Escritorio\SMARTFLOW\videos de prueba\data_video2 (1).mp4")  # For Video
 
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Detectar y configurar dispositivo (CPU/GPU)
device_str = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(device_str)
except Exception:
    pass
print(f"[INFO] PyTorch CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
try:
    print("[INFO] Modelo en:", next(model.model.parameters()).device)
except Exception:
    try:
        print("[INFO] Modelo en:", getattr(model, "device", device_str))
    except Exception:
        print("[INFO] Usando dispositivo:", device_str)
 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
 
 
 
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []
graphics_path = os.path.join(os.path.dirname(__file__), "graphics.png")
paused = False  # 'p' o espacio para pausar/reanudar, 'n' para avanzar 1 frame
 
while True:
    # Si está en pausa, espera tecla: 'n' avanza 1 frame, 'p'/espacio reanuda, 'q'/ESC sale
    if paused:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('p'), 32):  # 'p' o espacio
            paused = False
            continue
        if key == ord('n'):
            pass  # avanzar un frame (procesar una iteración)
        else:
            continue

    success, img = cap.read()
    if not success:
        break

    if os.path.isfile(graphics_path):
        try:
            imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
            if imgGraphics is not None:
                img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
        except Exception:
            pass
    # Mostrar dispositivo en la imagen
    cv2.putText(img, f"Device: {device_str.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Inferencia con dispositivo explícito
    results = model(img, stream=True, device=device_str)

 

    detections = np.empty((0, 5))
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
 
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
 
    resultsTracker = tracker.update(detections)
 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
 
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
 
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
 
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if key in (ord('p'), 32):  # 'p' o espacio
        paused = True
