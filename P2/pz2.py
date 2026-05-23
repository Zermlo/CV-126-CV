import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


source = "defaultCam.MOV"
cap = cv2.VideoCapture(source)
overlay = cv2.imread("default.jpg")
overlay_cap = cv2.VideoCapture("defaultCam.MOV")
#print(type(overlay_cap))
# стабилизация
last_screen = None
lost_frames = 0
MAX_LOST = 5
while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret2, overlay = overlay_cap.read()
    if not ret2: # тут видео на кадры раскладываем
        overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, overlay = overlay_cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screen_contour = None
    max_w = 0

    for cnt in contours: #Тут считаются контуры
        area = cv2.contourArea(cnt)
        if area < 30000:
            continue

        perimetr = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True) # По сути урощенный cnt

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)

            if 1.2 < aspect < 2.5: # смотрим, чтобы отношение сторон экрана было как у телевизора
                if w > max_w: #проверка по ширине
                    max_w = w
                    screen_contour = approx

    if screen_contour is not None: # стабилизация для того, чтобы видео не пропадало
        last_screen = screen_contour
        lost_frames = 0
    else:
        lost_frames += 1

    if last_screen is not None and lost_frames < MAX_LOST:
        screen_contour = last_screen
    else:
        screen_contour = None

    if screen_contour is not None: # Здесь происходит наложение на экран
        cv2.drawContours(frame, [screen_contour], -1, (0, 255, 0), 2)
        #print("screen_contour", screen_contour)
        screen_corners = order_points(screen_contour.reshape(4, 2).astype("float32")) #упорядочиваю точки
        #print("screen_corners", screen_corners)
        h, w = overlay.shape[0:2]

        src = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src, screen_corners) # строим перспективу

        warped = cv2.warpPerspective(overlay, M, (frame.shape[1], frame.shape[0])) # Искажение изображения под перспективу

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8) # Создаем маску
        cv2.fillConvexPoly(mask, screen_corners.astype(int), 255)

        inv_mask = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        overlay_fg = cv2.bitwise_and(warped, warped, mask=mask)

        frame = cv2.add(frame_bg, overlay_fg)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 640, 360)

    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("edges", 640, 360)

    cv2.imshow("result", frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('d'):
        continue
    elif key == ord('a'):
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2))

cap.release()
cv2.destroyAllWindows()