from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-5/weights/best.pt")
confidence = 0.6
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("video5368367111754256483.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    results = model(frame, conf=confidence)
    result_frame = results[0].plot()

    if len(results[0].boxes) > 0:
        cv2.imshow("Detection", result_frame)
    else:
        empty = frame.copy()
        cv2.putText(empty, "No detections", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Detection", empty)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()