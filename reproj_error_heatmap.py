import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from alike import ALike, configs
from demo_heatmap_mixed import (
    extract_orb_keypoints,
    tensor_from_bgr,
)

# ---------------- geometry utils ----------------

def compute_mean_reproj_error_and_inliers(kp1, kp2, matches, ransac_thresh=3.0):
    """
    Returns:
        mean_reproj_error (float | None)
        num_inliers (int)
    """
    if matches is None or len(matches) < 4:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None, 0

    inlier_mask = (mask.ravel() == 1)
    num_inliers = int(inlier_mask.sum())
    if num_inliers == 0:
        return None, 0

    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    proj = (H @ pts1_h.T).T
    proj = proj[:, :2] / proj[:, 2:]

    err = np.linalg.norm(proj - pts2, axis=1)
    return float(err[inlier_mask].mean()), num_inliers


# ---------------- ALike heatmap ----------------

@torch.no_grad()
def get_alike_heatmap(model, img_bgr):
    img_t = tensor_from_bgr(img_bgr)
    dense = model.extract_dense_map(img_t, ret_dict=True)
    score_map = dense["score_map"]
    return score_map[0, 0].detach().cpu().numpy()


# ---------------- ORB filtering ----------------

def filter_orb_by_heatmap(kps, desc, coords, heatmap, th):
    """
    Keeps ORB keypoints/descriptors whose (x,y) falls on a heatmap value >= th.
    """
    if desc is None or len(kps) == 0:
        return [], None

    H, W = heatmap.shape
    kp_f, des_f = [], []

    for i, (x, y) in enumerate(coords):
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H and heatmap[yi, xi] >= th:
            kp_f.append(kps[i])
            des_f.append(desc[i])

    if len(des_f) == 0:
        return [], None

    return kp_f, np.asarray(des_f)


# ---------------- model loading ----------------

def load_alike_model(model_name: str, model_path: str | None, device: str = "cpu"):
    cfg = configs[model_name].copy()
    cfg["device"] = device
    model = ALike(**cfg)
    model.eval()

    if model_path:
        mp = Path(model_path).expanduser()
        if mp.exists():
            state = torch.load(str(mp), map_location=device)
            # Accept both full checkpoints and raw state_dicts
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

            logging.info("Loaded model parameters from %s", str(mp))
            try:
                kb = os.path.getsize(mp) / 1024.0
                logging.info("Number of model parameters: %.3fKB", kb)
            except OSError:
                pass
        else:
            logging.warning("Model path not found: %s (continuing with default weights)", str(mp))

    return model


# ---------------- main experiment ----------------

def main(img1_path, img2_path, model_name="alike-t", model_path=None, nfeatures=5000, th=0.05, ransac_thresh=3.0):
    # load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one of the input images.")

    # load model
    model = load_alike_model(model_name=model_name, model_path=model_path, device="cpu")

    # ORB extraction
    coords1, kp1, des1 = extract_orb_keypoints(img1, nfeatures)
    coords2, kp2, des2 = extract_orb_keypoints(img2, nfeatures)

    # ALike heatmap (from first image)
    heatmap = get_alike_heatmap(model, img1)

    # filter ORB (only kp/des from image1 are filtered; image2 stays full ORB)
    kp1_f, des1_f = filter_orb_by_heatmap(kp1, des1, coords1, heatmap, th)

    # matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_orb = []
    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        matches_orb = bf.match(des1, des2)

    matches_filt = []
    if des1_f is not None and des2 is not None and len(des1_f) > 0 and len(des2) > 0:
        matches_filt = bf.match(des1_f, des2)

    # reprojection error + inliers
    err_orb, inl_orb = compute_mean_reproj_error_and_inliers(kp1, kp2, matches_orb, ransac_thresh=ransac_thresh)
    err_filt, inl_filt = compute_mean_reproj_error_and_inliers(kp1_f, kp2, matches_filt, ransac_thresh=ransac_thresh)

    print("\n=== Reprojection Error Experiment ===")
    print(f"ORB only        : {err_orb}")
    print(f"ORB ∩ ALike     : {err_filt}")
    print(f"Keypoints ORB   : {len(kp1)}")
    print(f"Keypoints filt  : {len(kp1_f)}")
    print(f"Geom inliers ORB: {inl_orb}")
    print(f"Geom inliers filt: {inl_filt}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("img1")
    parser.add_argument("img2")
    parser.add_argument("--model_name", default="alike-t")
    parser.add_argument(
        "--model_path",
        default=str(Path(__file__).resolve().parent / "models" / "alike-t.pth"),
        help="Path to ALike .pth weights (default: ./models/alike-t.pth next to this script)",
    )
    parser.add_argument("--nfeatures", type=int, default=5000)
    parser.add_argument("--th", type=float, default=0.05)
    parser.add_argument("--ransac", type=float, default=3.0, help="RANSAC reprojection threshold (pixels)")
    args = parser.parse_args()

    main(
        args.img1,
        args.img2,
        model_name=args.model_name,
        model_path=args.model_path,
        nfeatures=args.nfeatures,
        th=args.th,
        ransac_thresh=args.ransac,
    )
