import cv2
import numpy as np


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # левый верхний
    rect[2] = pts[np.argmax(s)]      # правый нижний

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # правый верхний
    rect[3] = pts[np.argmax(diff)]   # левый нижний

    return rect


def correct_perspective(img, bbox):
    src = order_points(bbox[0])

    width_a = np.linalg.norm(src[2] - src[3])
    width_b = np.linalg.norm(src[1] - src[0])
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(src[1] - src[2])
    height_b = np.linalg.norm(src[0] - src[3])
    max_height = max(int(height_a), int(height_b), 1)

    side = max(max_width, max_height)

    dst = np.float32([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (side, side))
    return warped


def decode_qr_code_from_camera():
    cap = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()

    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        img = cv2.flip(img, 1)

        data, bbox, _ = detector.detectAndDecode(img)

        view = img.copy()

        if bbox is not None and data:
            pts = bbox[0].astype(int)
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % len(pts)])
                cv2.line(view, pt1, pt2, (255, 0, 0), 3)

            corrected_img = correct_perspective(img, bbox)
            corrected_data, corrected_bbox, _ = detector.detectAndDecode(corrected_img)

            cv2.imshow("Corrected QR", corrected_img)

            if corrected_data:
                cv2.putText(view, f"Corrected: {corrected_data}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(view, "Corrected: not decoded", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if data:
            print(f"Decoded Data: {data}")
            cv2.putText(view, f"Raw: {data}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(view, "Raw: not decoded", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("QR Code Detection", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


decode_qr_code_from_camera()