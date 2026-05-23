import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

scale = 0.5

points = []
line_ready = False

def mouse(event, x, y, flags, param):
    global points, line_ready

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((int(x / scale), int(y / scale)))
        if len(points) == 2:
            line_ready = True


cap = cv2.VideoCapture("8.avi")

ret, frame = cap.read()
if not ret:
    exit()

cv2.namedWindow("set line")
cv2.setMouseCallback("set line", mouse)

while True:
    show = cv2.resize(frame, None, fx=scale, fy=scale)

    img = show.copy()

    if len(points) > 0:
        cv2.circle(img, (int(points[0][0] * scale), int(points[0][1] * scale)), 5, (0, 0, 255), -1)

    if len(points) == 2:
        cv2.line(img, (int(points[0][0] * scale), int(points[0][1] * scale)), (int(points[1][0] * scale), int(points[1][1] * scale)), (0, 255, 0), 2)

    cv2.imshow("set line", img)

    if cv2.waitKey(1) & 0xFF == 27 or line_ready:
        break

cv2.destroyWindow("set line")

(x1, y1), (x2, y2) = points

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

counted = set()
prev_pos = {}
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > 0.5:
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
            w = x2b - x1b
            h = y2b - y1b
            detections.append(([x1b, y1b, w, h], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        a1, b1, a2, b2 = map(int, t.to_ltrb())

        cx = (a1 + a2) // 2
        cy = b2
        line_side = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)

        if track_id in prev_pos:
            prev_side = prev_pos[track_id]

            if track_id not in counted and prev_side * line_side < 0:
                counted.add(track_id)
                count += 1

        prev_pos[track_id] = line_side

        cv2.rectangle(frame, (a1, b1), (a2, b2), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(track_id), (a1, b1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    show = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow("result", show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()