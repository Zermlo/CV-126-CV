import cv2
import numpy as np

SAVE_PATH = "stereo_calibration.npz"
videos = ["1943352024.mp4","1946402024.mp4","1957512024.mp4"]
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.035
MAX_CALIBRATION_FRAMES = 80
MIN_FRAME_GAP = 25

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_left = []
imgpoints_right = []

image_size = None
frame_id = 0
last_saved_frame = 0
last_frame = 0
#zapas = 30
propusk = 30

for path in videos:
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % propusk != 0:
            continue
        h, w = frame.shape[:2]
        left_frame = frame[:, :w // 2]
        right_frame = frame[:, w // 2:]

        left_grey = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        image_size = right_grey.shape[::-1]

        ret_left, corners_left = cv2.findChessboardCorners(left_grey, CHECKERBOARD)
        ret_right, corners_right = cv2.findChessboardCorners(right_grey, CHECKERBOARD)
        #if ret_left and ret_right and last_frame + zapas <= frame_id:
        if ret_left and ret_right: # and last_frame + zapas <= frame_id:
            corners_left = cv2.cornerSubPix(
                left_grey, corners_left, (11, 11), (-1, -1), criteria
            )
            imgpoints_left.append(corners_left)

            corners_right = cv2.cornerSubPix(
                right_grey, corners_right, (11, 11), (-1, -1), criteria
            )
            imgpoints_right.append(corners_right)
            objpoints.append(objp.copy())
            last_frame = frame_id

        if ret_left:
            cv2.drawChessboardCorners(left_frame, CHECKERBOARD, corners_left, ret_left)

        if ret_right:
            cv2.drawChessboardCorners(right_frame, CHECKERBOARD, corners_right, ret_right)

        preview = np.hstack((left_frame, right_frame))

        cv2.putText(
            preview,
            f"Frames: {len(objpoints)} / {MAX_CALIBRATION_FRAMES}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("Stereo calibration", preview)

        if len(objpoints) >= MAX_CALIBRATION_FRAMES:
            print("Набрано достаточно кадров")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Всего кадров для калибровки:", len(objpoints))

if len(objpoints) < 8:
    print("Слишком мало кадров. Нужно хотя бы 8 успешных кадров.")
    exit()

print("Калибровка левой камеры...")
ret_l, K1, D1, _, _ = cv2.calibrateCamera(
    objpoints,
    imgpoints_left,
    image_size,
    None,
    None
)

print("Калибровка правой камеры...")
ret_r, K2, D2, _, _ = cv2.calibrateCamera(
    objpoints,
    imgpoints_right,
    image_size,
    None,
    None
)

print("Стереокалибровка...")
ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1,
    D1,
    K2,
    D2,
    image_size,
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1,
    D1,
    K2,
    D2,
    image_size,
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=1
)

np.savez(
    SAVE_PATH,
    K1=K1,
    D1=D1,
    K2=K2,
    D2=D2,
    R=R,
    T=T,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q,
    image_size=image_size
)

print("Готово")
print("Stereo calibration saved to:", SAVE_PATH)
print("Left calibration error:", ret_l)
print("Right calibration error:", ret_r)
print("Stereo calibration error:", ret_stereo)
print("T:", T.ravel())