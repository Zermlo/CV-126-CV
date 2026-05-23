import os
import cv2
import random
video_path = "video5368367111754256483.mp4"
output_dir = "images/"

n = 0
cap = cv2.VideoCapture(video_path)
frame_count = 0
print(os.path.abspath(output_dir))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if frame_count % 5 == 0:
        cv2.imwrite(f"{output_dir}/frame_{n}.jpg", frame)
        n += 1
    frame_count += 1

cap.release()