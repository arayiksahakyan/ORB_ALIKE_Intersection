#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import logging
import numpy as np

from alike import ALike, configs  # важно: alike.py и models уже настроены под CPU/heatmap


# ---------- ALike utils ----------

def tensor_from_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR OpenCV image to torch tensor (1,3,H,W) in [0,1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).to(torch.float32) / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
    return t


def to_heatmap(score_map: torch.Tensor) -> np.ndarray:
    """
    score_map: (1,1,H,W) tensor -> (H,W,3) BGR heatmap
    """
    sm = score_map[0, 0].detach().cpu().numpy()  # (H,W)
    sm = np.clip(sm, 0.0, 1.0)
    heat = (sm * 255.0).astype('uint8')
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat_bgr


# ---------- ORB utils ----------

def extract_orb_keypoints(img_bgr: np.ndarray, nfeatures: int = 1000):
    """
    Extract ORB keypoints and descriptors from a BGR frame.

    Returns:
        coords: (N,2) float32 array with (x, y)
        keypoints: list[cv2.KeyPoint]
        descriptors: np.ndarray or None
    """
    if img_bgr is None:
        raise ValueError("Input frame is None in extract_orb_keypoints")

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img_bgr, None)
    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return coords, keypoints, descriptors


# ---------- run on image ----------

def run_on_image(model: ALike, path: str, nfeatures: int):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    # ALike heatmap
    inp = tensor_from_bgr(img).to(model.device)
    dense = model.extract_dense_map(inp, ret_dict=True)
    score_map = dense['score_map']           # (1,1,H,W)
    score_np = score_map[0, 0].detach().cpu().numpy()
    heatmap = to_heatmap(score_map)          # (H,W,3) BGR

    # ORB keypoints (на оригинальном кадре, но рисуем на heatmap)
    coords, kps, desc = extract_orb_keypoints(img, nfeatures=nfeatures)
    
    vis = img.copy()   # ← визуализация на ОРИГИНАЛЬНОМ кадре
    print(f"[IMAGE] ORB keypoints: {len(coords)}")

    HEATMAP_TH = 0.05
    H, W = score_np.shape
    filtered_coords = []

    # 🔴 все ORB
    for (x, y) in coords:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    # 🟢 ORB ∩ ALike
    for (x, y) in coords:
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H:
            if score_np[yi, xi] >= HEATMAP_TH:
                filtered_coords.append((xi, yi))
                cv2.circle(vis, (xi, yi), 2, (0, 255, 0), -1)

    print(f"[IMAGE] ORB total: {len(coords)}")
    print(f"[IMAGE] ORB ∩ ALike: {len(filtered_coords)}")

    cv2.imshow("ALike heatmap + ORB", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- run on video ----------

def run_on_video(model: ALike, path: str, nfeatures: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info("Press 'q' to quit video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("ALike heatmap + ORB (video)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ALike heatmap + ORB (video)", w, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        inp = tensor_from_bgr(frame).to(model.device)
        dense = model.extract_dense_map(inp, ret_dict=True)
        score_map = dense['score_map']

        score_np = score_map[0, 0].detach().cpu().numpy()
        heatmap = to_heatmap(score_map)

        coords, kps, desc = extract_orb_keypoints(frame, nfeatures=nfeatures)
        vis = frame.copy()

        HEATMAP_TH = 0.05
        H, W = score_np.shape
        filtered_coords = []

        for (x, y) in coords:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

        for (x, y) in coords:
            xi, yi = int(x), int(y)
            if 0 <= xi < W and 0 <= yi < H:
                if score_np[yi, xi] >= HEATMAP_TH:
                    filtered_coords.append((xi, yi))
                    cv2.circle(vis, (xi, yi), 2, (0, 255, 0), -1)

        cv2.imshow("ALike heatmap + ORB (video)", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"]


# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALike heatmap + ORB keypoints")
    parser.add_argument("input", type=str, help="path to image or video file")
    parser.add_argument("--model", choices=list(configs.keys()), default="alike-t")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nfeatures", type=int, default=1000,
                        help="number of ORB keypoints to detect")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # грузим ALike
    cfg = configs[args.model].copy()
    cfg["device"] = args.device
    model = ALike(**cfg)
    model.eval()

    path = args.input
    if not os.path.exists(path):
        raise SystemExit(f"Input path does not exist: {path}")

    if is_image_file(path):
        run_on_image(model, path, nfeatures=args.nfeatures)
    else:
        run_on_video(model, path, nfeatures=args.nfeatures)

