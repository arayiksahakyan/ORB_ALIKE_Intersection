#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import logging
import numpy as np

from alike import ALike, configs


# ---------- ALike utils ----------

def tensor_from_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR OpenCV image to torch tensor (1,3,H,W) in [0,1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).to(torch.float32) / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t


def to_heatmap(score_map: torch.Tensor) -> np.ndarray:
    """
    score_map: (1,1,H,W) tensor -> (H,W,3) BGR heatmap
    """
    sm = score_map[0, 0].detach().cpu().numpy()
    sm = np.clip(sm, 0.0, 1.0)
    heat = (sm * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat_bgr


# ---------- BEBLID utils ----------

def create_beblid(scale_factor: float = 1.0, n_bits: int = 101):
    if not hasattr(cv2, "xfeatures2d"):
        raise RuntimeError("cv2.xfeatures2d not found. Install opencv-contrib-python.")

    if not hasattr(cv2.xfeatures2d, "BEBLID_create"):
        raise RuntimeError("cv2.xfeatures2d.BEBLID_create not found. Install opencv-contrib-python.")

    return cv2.xfeatures2d.BEBLID_create(scale_factor, n_bits)


# ---------- ORB + filter + BEBLID ----------

def detect_orb_keypoints(img_bgr: np.ndarray, nfeatures: int = 1000):
    """
    Detect ORB keypoints only.
    """
    if img_bgr is None:
        raise ValueError("Input frame is None in detect_orb_keypoints")

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints = orb.detect(img_bgr, None)
    return keypoints


def filter_keypoints_by_heatmap(keypoints, score_np: np.ndarray, heatmap_th: float):
    """
    Keep only those keypoints that fall into strong ALike heatmap zones.
    """
    h, w = score_np.shape
    filtered = []

    for kp in keypoints:
        x, y = kp.pt
        xi, yi = int(round(x)), int(round(y))

        if 0 <= xi < w and 0 <= yi < h:
            if score_np[yi, xi] >= heatmap_th:
                filtered.append(kp)

    return filtered


def compute_beblid_descriptors(
    img_bgr: np.ndarray,
    keypoints,
    scale_factor: float = 1.0,
    n_bits: int = 101
):
    """
    Compute BEBLID descriptors for already detected keypoints.
    """
    if len(keypoints) == 0:
        return [], None

    beblid = create_beblid(scale_factor=scale_factor, n_bits=n_bits)
    keypoints, descriptors = beblid.compute(img_bgr, keypoints)
    return keypoints, descriptors


def keypoints_to_coords(keypoints):
    if len(keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def extract_orb_beblid_with_alike_filter(
    img_bgr: np.ndarray,
    score_np: np.ndarray,
    nfeatures: int = 1000,
    heatmap_th: float = 0.05,
    beblid_scale_factor: float = 1.0,
    beblid_n_bits: int = 101
):
    all_keypoints = detect_orb_keypoints(img_bgr, nfeatures=nfeatures)
    filtered_keypoints = filter_keypoints_by_heatmap(all_keypoints, score_np, heatmap_th)

    filtered_keypoints, descriptors = compute_beblid_descriptors(
        img_bgr,
        filtered_keypoints,
        scale_factor=beblid_scale_factor,
        n_bits=beblid_n_bits
    )

    all_coords = keypoints_to_coords(all_keypoints)
    filtered_coords = keypoints_to_coords(filtered_keypoints)

    return {
        "all_keypoints": all_keypoints,
        "all_coords": all_coords,
        "filtered_keypoints": filtered_keypoints,
        "filtered_coords": filtered_coords,
        "descriptors": descriptors
    }


def draw_keypoints_overlay(img_bgr: np.ndarray, all_keypoints, filtered_keypoints):
    vis = img_bgr.copy()

    for kp in all_keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

    for kp in filtered_keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

    return vis


# ---------- run on image ----------

def run_on_image(
    model: ALike,
    path: str,
    nfeatures: int,
    heatmap_th: float,
    beblid_scale_factor: float,
    beblid_n_bits: int
):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    inp = tensor_from_bgr(img).to(model.device)
    dense = model.extract_dense_map(inp, ret_dict=True)
    score_map = dense["score_map"]
    score_np = score_map[0, 0].detach().cpu().numpy()
    _ = to_heatmap(score_map)

    result = extract_orb_beblid_with_alike_filter(
        img_bgr=img,
        score_np=score_np,
        nfeatures=nfeatures,
        heatmap_th=heatmap_th,
        beblid_scale_factor=beblid_scale_factor,
        beblid_n_bits=beblid_n_bits
    )

    vis = draw_keypoints_overlay(
        img,
        result["all_keypoints"],
        result["filtered_keypoints"]
    )

    desc = result["descriptors"]
    desc_shape = None if desc is None else desc.shape

    print(f"[IMAGE] ORB total: {len(result['all_keypoints'])}")
    print(f"[IMAGE] ORB ∩ ALike: {len(result['filtered_keypoints'])}")
    print(f"[IMAGE] BEBLID descriptors shape: {desc_shape}")

    cv2.namedWindow("ALike + ORB detector + BEBLID descriptor", cv2.WINDOW_NORMAL)
    cv2.imshow("ALike + ORB detector + BEBLID descriptor", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- run on video ----------

def run_on_video(
    model: ALike,
    path: str,
    nfeatures: int,
    heatmap_th: float,
    beblid_scale_factor: float,
    beblid_n_bits: int
):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info("Press 'q' to quit video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_name = "ALike + ORB detector + BEBLID descriptor (video)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        inp = tensor_from_bgr(frame).to(model.device)
        dense = model.extract_dense_map(inp, ret_dict=True)
        score_map = dense["score_map"]
        score_np = score_map[0, 0].detach().cpu().numpy()
        _ = to_heatmap(score_map)

        result = extract_orb_beblid_with_alike_filter(
            img_bgr=frame,
            score_np=score_np,
            nfeatures=nfeatures,
            heatmap_th=heatmap_th,
            beblid_scale_factor=beblid_scale_factor,
            beblid_n_bits=beblid_n_bits
        )

        vis = draw_keypoints_overlay(
            frame,
            result["all_keypoints"],
            result["filtered_keypoints"]
        )

        desc = result["descriptors"]
        desc_count = 0 if desc is None else len(desc)

        print(f"ORB total: {len(result['all_keypoints'])}")
        print(f"ORB ∩ ALike: {len(result['filtered_keypoints'])}")
        print(f"BEBLID descriptors: {desc_count}")

        cv2.imshow(window_name, vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"]


# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALike heatmap + ORB detector + BEBLID descriptor"
    )
    parser.add_argument("input", type=str, help="path to image or video file")
    parser.add_argument("--model", choices=list(configs.keys()), default="alike-t")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nfeatures", type=int, default=1000,
                        help="number of ORB keypoints to detect")
    parser.add_argument("--heatmap_th", type=float, default=0.05,
                        help="ALike score threshold for filtering ORB keypoints")
    parser.add_argument("--beblid_scale_factor", type=float, default=1.0,
                        help="BEBLID scale factor")
    parser.add_argument("--beblid_n_bits", type=int, default=101,
                        choices=[100, 101, 102, 103],
                        help="BEBLID descriptor size mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = configs[args.model].copy()
    cfg["device"] = args.device
    model = ALike(**cfg)
    model.eval()

    path = args.input
    if not os.path.exists(path):
        raise SystemExit(f"Input path does not exist: {path}")

    if is_image_file(path):
        run_on_image(
            model=model,
            path=path,
            nfeatures=args.nfeatures,
            heatmap_th=args.heatmap_th,
            beblid_scale_factor=args.beblid_scale_factor,
            beblid_n_bits=args.beblid_n_bits
        )
    else:
        run_on_video(
            model=model,
            path=path,
            nfeatures=args.nfeatures,
            heatmap_th=args.heatmap_th,
            beblid_scale_factor=args.beblid_scale_factor,
            beblid_n_bits=args.beblid_n_bits
        )
