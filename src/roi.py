"""
Étape 2 : couleur moyenne (R, G, B) de 3 zones de peau à chaque image,
enregistrée pendant DURATION_S secondes dans data/signal_webcam.npz
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions

MODEL_PATH = "models/face_landmarker.task"
OUT_PATH = "data/signal_webcam.npz"
ROI_KEY_POINTS = (151, 50, 280)  # front, joue gauche écran, joue droite écran
ROI_SIZE = 30
DURATION_S = 30

if os.path.exists(MODEL_PATH):
    print(f"Model OK ({os.path.getsize(MODEL_PATH) / 1e6:.2f} Mo)")
else:
    exit("Model NOT FOUND - lance le script depuis la racine du projet")

opts = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

times = []   # temps de chaque mesure (s)
colors = []  # pour chaque mesure : 3 zones x (R, G, B)

with vision.FaceLandmarker.create_from_options(opts) as landmarker:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        exit("Camera not found")
    fps_announced = cap.get(cv2.CAP_PROP_FPS)

    t0 = time.perf_counter()
    n_frames = 0
    fps_real = 0.0
    green_forehead = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed_s = time.perf_counter() - t0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(elapsed_s * 1000))

        if result.face_landmarks:
            h, w = frame.shape[:2]
            landmarks = result.face_landmarks[0]
            zones = []
            for i in ROI_KEY_POINTS:
                p = landmarks[i]
                x_px, y_px = int(p.x * w), int(p.y * h)
                x1 = max(0, x_px - ROI_SIZE // 2)
                y1 = max(0, y_px - ROI_SIZE // 2)
                x2 = min(w, x_px + ROI_SIZE // 2)
                y2 = min(h, y_px + ROI_SIZE // 2)

                patch = rgb[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                zones.append(patch.mean(axis=(0, 1)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if len(zones) == len(ROI_KEY_POINTS):
                times.append(elapsed_s)
                colors.append(zones)
                green_forehead = zones[0][1]
        else:
            cv2.putText(frame, "No face", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        n_frames += 1
        if elapsed_s > 0:
            fps_real = n_frames / elapsed_s
        cv2.putText(frame, f"FPS: {fps_real:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"G front: {green_forehead:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"REC {elapsed_s:.0f}/{DURATION_S} s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("roi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or elapsed_s >= DURATION_S:
            break

    cap.release()
    cv2.destroyAllWindows()

if not times:
    exit("Aucune mesure : pas de visage détecté")

t = np.array(times)
signal_rgb = np.array(colors)  # forme (N, 3 zones, 3 couleurs)
os.makedirs("data", exist_ok=True)
np.savez(OUT_PATH, t=t, rgb=signal_rgb, roi_points=np.array(ROI_KEY_POINTS))

duration = t[-1] - t[0]
print(f"t   : {t.shape}")
print(f"rgb : {signal_rgb.shape}")
print(f"Durée = {duration:.1f} s | FPS mesures = {len(t) / duration:.2f} | "
      f"FPS annoncé = {fps_announced:.2f}")
print(f"Sauvegardé dans {OUT_PATH}")