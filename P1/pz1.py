import sys
import cv2
import tkinter as tk
from threading import Thread
import random

boxes = []
box_size = 15

def add_box(evt, x, y, flags, param):
    global box_size
    r = random.randint(1,20)
    if evt == cv2.EVENT_LBUTTONDOWN:
        boxes.append((x - box_size - r, y - box_size - r, x + box_size + r, y + box_size + r))

def video_loop():
    global boxes
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", add_box)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Camera", 140, 360)

    while 0 == 0:
        ret, frame = cap.read()
        if not ret:
            break

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('c'), ord('C')):
            print("test1")
            boxes.clear()

        if key in (ord('q'), ord('Q')):
            #cap.release()
            #cv2.destroyAllWindows()
            break

        if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            #stop_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()


def start_video():
    Thread(target=video_loop, daemon=True).start()


def create_ui():
    root = tk.Tk()
    root.title("Control")

    root.geometry("300x200")

    tk.Button(root, text="Запустить видео", command=start_video).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    #src = sys.argv[1] if len(sys.argv) > 1 else "0"
    #if src.isdigit():
    #    src = int(src)
    create_ui()