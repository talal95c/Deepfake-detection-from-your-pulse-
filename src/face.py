"""
Webcam + 478 points du visage + FPS réel
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL_PATH = "models/face_landmarker.task"
KEY_POINTS = (10, 151, 50, 117, 280, 346)

if os.path.exists(MODEL_PATH):
    print(f"Model OK ({os.path.getsize(MODEL_PATH) / 1e6:.2f} Mo)")
else:
    exit("Model NOT FOUND - lance le script depuis la racine du projet")

opts = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

with vision.FaceLandmarker.create_from_options(opts) as landmarker:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        exit("Camera not found")
    fps_announced = cap.get(cv2.CAP_PROP_FPS)

    t0 = time.perf_counter()
    n_frames = 0
    fps_real = 0.0

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
            for i, p in enumerate(result.face_landmarks[0]):
                x_px, y_px = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)
                if i in KEY_POINTS:
                    cv2.putText(frame, str(i), (x_px, y_px),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "No face", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        n_frames += 1
        if elapsed_s > 0:
            fps_real = n_frames / elapsed_s
        cv2.putText(frame, f"FPS: {fps_real:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print(f"FPS annoncé = {fps_announced:.2f} | FPS réel = {fps_real:.2f}")