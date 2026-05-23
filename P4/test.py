from ultralytics import YOLO
import os
import shutil
import random

def perenos(file_list, img_dst, lbl_dst):
    for image in file_list:
        llabel = os.path.splitext(image)[0]

        img_src_path = os.path.join(images_dir, image)
        lbl_src_path = os.path.join(labels_dir, llabel + ".txt")

        if not (os.path.exists(img_src_path) and os.path.exists(lbl_src_path)):
            print(f"Пропущено. нет пары для {image}")
            continue

        img_dst_path = os.path.join(img_dst, image)
        lbl_dst_path = os.path.join(lbl_dst, llabel + ".txt")

        shutil.copy2(img_src_path, img_dst_path)
        shutil.copy2(lbl_src_path, lbl_dst_path)

split_ratio_train = 0.8
split_ratio_val = 0.1
split_ratio_test = 0.1

images_dir = "dataset/images"
labels_dir = "dataset/labels"

train_images_dir = "dataset/images/train"
val_images_dir = "dataset/images/val"
test_images_dir = "dataset/images/test"

train_labels_dir = "dataset/labels/train"
val_labels_dir = "dataset/labels/val"
test_labels_dir = "dataset/labels/test"


image_files = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(".jpg")
]

random.shuffle(image_files)

total = len(image_files)

train_end = int(total * split_ratio_train)
val_end = train_end + int(total * split_ratio_val)

train_files = image_files[:train_end]
valid_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

perenos(train_files, train_images_dir, train_labels_dir)
perenos(valid_files, val_images_dir, val_labels_dir)
perenos(test_files, test_images_dir, test_labels_dir)

model = YOLO("yolov8n.pt")

# data = {
#     "path": "dataset",
#     "train": "images/train",
#     "val": "images/val",
#     "names": {
#         0: "robot"
#     }
# }

model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)