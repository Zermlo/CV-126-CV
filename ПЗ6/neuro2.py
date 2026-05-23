import os
import sys
import argparse

import cv2
import numpy as np
import torch

# Путь к официальному репозиторию IGEV-Stereo
IGEV_ROOT = "IGEV-Stereo"
sys.path.append(os.path.join(IGEV_ROOT, "core"))

from core.igev_stereo import IGEVStereo
from utils.utils import InputPadder


def load_calibration(npz_path: str):
    calib = np.load(npz_path)

    K1 = calib["K1"]
    D1 = calib["D1"]
    K2 = calib["K2"]
    D2 = calib["D2"]
    R1 = calib["R1"]
    R2 = calib["R2"]
    P1 = calib["P1"]
    P2 = calib["P2"]

    image_size = tuple(calib["image_size"].astype(int))  # (w, h)

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_16SC2
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_16SC2
    )

    return image_size, (map1x, map1y), (map2x, map2y)


def rectify(img, maps):
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)


def build_model(ckpt_path: str, device: str):
    # Аргументы повторяют официальный demo_video.py
    model_args = argparse.Namespace(
        restore_ckpt=ckpt_path,
        save_numpy=False,
        left_imgs="",
        right_imgs="",
        mixed_precision=True,
        precision_dtype="float32",
        valid_iters=16,
        hidden_dims=[128] * 3,
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        slow_fast_gru=False,
        n_gru_layers=3,
        max_disp=192,
    )

    model = torch.nn.DataParallel(IGEVStereo(model_args), device_ids=[0])
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.module.to(device)
    model.eval()
    return model


def infer_disparity(model, left_bgr, right_bgr, device="cuda"):
    # В официальном демо изображения идут как RGB
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    image1 = torch.from_numpy(left_rgb).permute(2, 0, 1).float()[None].to(device)
    image2 = torch.from_numpy(right_rgb).permute(2, 0, 1).float()[None].to(device)

    padder = InputPadder(image1.shape, divis_by=32)
    image1_pad, image2_pad = padder.pad(image1, image2)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            disp = model(image1_pad, image2_pad, iters=16, test_mode=True)

    disp = padder.unpad(disp)
    disp = disp.squeeze().detach().cpu().numpy()

    # Только для отображения
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)

    return disp_color, disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="1954522024.mp4")
    parser.add_argument("--calib", default="stereo_calibration.npz")
    parser.add_argument(
        "--ckpt",
        default=os.path.join(IGEV_ROOT, "pretrained_models/sceneflow/sceneflow.pth"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Для IGEV-Stereo здесь нужен CUDA/GPU.")

    device = "cuda"

    image_size, left_maps, right_maps = load_calibration(args.calib)
    model = build_model(args.ckpt, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        left = frame[:, :w // 2]
        right = frame[:, w // 2:]

        # Если размер кадра не совпадает с калибровкой, приводим к нужному
        if (left.shape[1], left.shape[0]) != image_size:
            left = cv2.resize(left, image_size)
            right = cv2.resize(right, image_size)

        left = rectify(left, left_maps)
        right = rectify(right, right_maps)

        disp_color, _ = infer_disparity(model, left, right, device=device)

        cv2.imshow("Left", left)
        cv2.imshow("Right", right)
        cv2.imshow("IGEV Disparity", disp_color)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()