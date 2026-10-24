import os 
import cv2 
import mediapipe as mp 
import numpy as np 
import scipy 



print(f"OpenCV Version: {cv2.__version__}")
print(f"mediapipe Version: {mp.__version__}")
print(f"Numpy Version:{np.__version__}")
print(f"Scipy Version:{scipy.__version__}")

model_path = "models/face_landmarker.task"
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"Model OK ({size_mb:.2f} Mo)")
else:
    print("Model NOT FOUND - lance le script depuis la racine du projet")

cap = cv2.VideoCapture(0,cv2.CAP_MSMF)
if cap.isOpened() == False:
    exit("Camera not found")
else :
    print("Camera found")


ok, frame = cap.read()
if not ok:
    cap.release()
    exit("Camera opened but no frame read")
print(f"Frame read: {ok}  shape = {frame.shape}")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")
cap.release()