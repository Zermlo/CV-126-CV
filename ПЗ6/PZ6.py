import numpy as np
import cv2
#Это Block Matching
# Загрузка калибровки из npz
calib = np.load("stereo_calibration.npz")

K1 = calib["K1"]
D1 = calib["D1"]
K2 = calib["K2"]
D2 = calib["D2"]
R1 = calib["R1"]
R2 = calib["R2"]
P1 = calib["P1"]
P2 = calib["P2"]
Q = calib["Q"]
image_size = tuple(calib["image_size"].astype(int))

map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

def rectify_left(img):
    return cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR)

def rectify_right(img):
    return cv2.remap(img, map2x, map2y, cv2.INTER_LINEAR)

# StereoBM создаём один раз
stereo = cv2.StereoBM_create(
    numDisparities=16 * 6,
    blockSize=15
)

def testim(left, right):
    imgL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity

cap = cv2.VideoCapture("1954522024.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    # Rectification
    left = rectify_left(left)
    right = rectify_right(right)

    # Карта глубины
    disparity = testim(left, right)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    cv2.imshow("Disparity", disparity)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()