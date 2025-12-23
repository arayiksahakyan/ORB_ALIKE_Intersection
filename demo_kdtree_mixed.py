import os
import cv2
import torch
import argparse
import logging
import numpy as np
from scipy.spatial import cKDTree

from alike import ALike, configs  # modified ALike for cpu


# ---------- ALike utils ----------

def tensor_from_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR OpenCV image to torch tensor (1,3,H,W) in [0,1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).to(torch.float32) / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
    return t


def extract_alike_keypoints(score_map: torch.Tensor,
                            top_k: int = 2000) -> np.ndarray:
    """
    Extract sparse ALike keypoints from dense score_map by taking top_k maxima.

    Args:
        score_map: torch tensor of shape (1,1,H,W)
        top_k: number of strongest locations to keep

    Returns:
        pts: (M,2) float32 array of (x, y) coordinates in image pixels.
    """
    sm = score_map[0, 0].detach().cpu().numpy()  # (H,W)
    H, W = sm.shape
    flat = sm.ravel()

    top_k = min(top_k, flat.size)
    if top_k <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    idx = np.argpartition(-flat, top_k - 1)[:top_k]
    ys, xs = np.unravel_index(idx, (H, W))
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (M,2) with (x,y)

    return pts


# ---------- ORB utils ----------

def extract_orb_keypoints(img_bgr: np.ndarray, nfeatures: int = 50000):
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

    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32), [], descriptors

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return coords, keypoints, descriptors


# ---------- KDTree ----------

def match_kpts_kdtree(orb_coords: np.ndarray,
                      alike_coords: np.ndarray,
                      radius: float = 3.0) -> np.ndarray:
    """
    Match ORB keypoints with ALike keypoints using cKDTree
    """
    N = len(orb_coords)
    mask = np.zeros(N, dtype=bool)

    if N == 0 or len(alike_coords) == 0:
        return mask

    tree = cKDTree(alike_coords)
    dists, _ = tree.query(orb_coords, distance_upper_bound=radius)
    mask = np.isfinite(dists)
    return mask


# ---------- drawing ----------

def draw_orb_matches_on_frame(frame_bgr: np.ndarray,
                              orb_coords: np.ndarray,
                              match_mask: np.ndarray) -> np.ndarray:
    """
    Draw ORB keypoints on the input frame:
      - matched (intersection with ALike) in GREEN
      - unmatched in RED
    """
    vis = frame_bgr.copy()
    for (x, y), is_match in zip(orb_coords, match_mask):
        if is_match:
            color = (0, 255, 0)   # green
        else:
            color = (0, 0, 255)   # red
        cv2.circle(vis, (int(x), int(y)), 2, color, -1)
    return vis


# ---------- run on image ----------

def run_on_image(model: ALike, path: str, nfeatures: int,
                 alike_top_k: int = 2000,
                 match_radius: float = 3.0):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    # ALike dense score map
    inp = tensor_from_bgr(img).to(model.device)
    dense = model.extract_dense_map(inp, ret_dict=True)
    score_map = dense['score_map']           # (1,1,H,W)

    # ALike sparse keypoints from score map
    alike_pts = extract_alike_keypoints(score_map, top_k=alike_top_k)

    # ORB keypoints on original image
    orb_coords, kps, desc = extract_orb_keypoints(img, nfeatures=nfeatures)
    logging.info(f"[IMAGE] ORB keypoints: {len(orb_coords)}, ALike points: {len(alike_pts)}")

    # KDTree intersection
    match_mask = match_kpts_kdtree(orb_coords, alike_pts, radius=match_radius)

    # draw on input image
    vis = draw_orb_matches_on_frame(img, orb_coords, match_mask)

    cv2.namedWindow("ORB vs ALike (IMAGE)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("ORB vs ALike (IMAGE)", vis.shape[1], vis.shape[0])
    cv2.imshow("ORB vs ALike (IMAGE)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- run on video ----------

def run_on_video(model: ALike, path: str, nfeatures: int,
                 alike_top_k: int = 2000,
                 match_radius: float = 3.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info("Press 'q' to quit video")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        # ALike dense score map
        inp = tensor_from_bgr(frame).to(model.device)
        dense = model.extract_dense_map(inp, ret_dict=True)
        score_map = dense['score_map']

        # ALike sparse keypoints
        alike_pts = extract_alike_keypoints(score_map, top_k=alike_top_k)

        # ORB keypoints
        orb_coords, kps, desc = extract_orb_keypoints(frame, nfeatures=nfeatures)

        # KDTree intersection
        match_mask = match_kpts_kdtree(orb_coords, alike_pts, radius=match_radius)

        # draw on input frame
        vis = draw_orb_matches_on_frame(frame, orb_coords, match_mask)

        cv2.namedWindow("ORB vs ALike (VIDEO)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("ORB vs ALike (VIDEO)", vis.shape[1], vis.shape[0])
        cv2.imshow("ORB vs ALike (VIDEO)", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- helpers ----------

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"]


# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALike dense map + ORB keypoints + KDTree intersection"
    )
    parser.add_argument("input", type=str, help="path to image or video file")
    parser.add_argument("--model", choices=list(configs.keys()), default="alike-t")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nfeatures", type=int, default=5000,
                        help="number of ORB keypoints to detect")
    parser.add_argument("--alike-topk", type=int, default=2000,
                        help="number of strongest ALike points from score map")
    parser.add_argument("--match-radius", type=float, default=3.0,
                        help="max distance in pixels for ORB–ALike intersection")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # load ALike
    cfg = configs[args.model].copy()
    cfg["device"] = args.device
    model = ALike(**cfg)
    model.eval()

    path = args.input
    if not os.path.exists(path):
        raise SystemExit(f"Input path does not exist: {path}")

    if is_image_file(path):
        run_on_image(
            model,
            path,
            nfeatures=args.nfeatures,
            alike_top_k=args.alike_topk,
            match_radius=args.match_radius,
        )
    else:
        run_on_video(
            model,
            path,
            nfeatures=args.nfeatures,
            alike_top_k=args.alike_topk,
            match_radius=args.match_radius,
        )
