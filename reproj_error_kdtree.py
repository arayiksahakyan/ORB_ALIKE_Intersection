#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import logging

from alike import ALike, configs

# берем ровно те же функции, что использует demo_heatmap_mixed.py
from demo_heatmap_mixed import (
    tensor_from_bgr,
    extract_orb_keypoints,
    extract_alike_keypoints,
    match_kpts_kdtree,
)


# ---------------- geometry utils ----------------

def compute_mean_reproj_error(kp1, kp2, matches, ransac_thresh=3.0):
    """
    Compute mean reprojection error in pixels using homography (RANSAC).
    Returns None if not enough matches or homography fails.
    """
    if matches is None or len(matches) < 4:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None

    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return None

    pts1_in = pts1[inl]
    pts2_in = pts2[inl]

    pts1_h = cv2.convertPointsToHomogeneous(pts1_in)[:, 0, :]   # (N,3)
    proj = (H @ pts1_h.T).T                                     # (N,3)
    proj = proj[:, :2] / proj[:, 2:3]                           # (N,2)

    err = np.linalg.norm(proj - pts2_in, axis=1)
    return float(err.mean())


# ---------------- helpers ----------------

def filter_orb_by_mask(kps, desc, mask_bool):
    """
    Keep only kps/desc where mask_bool=True.
    """
    idx = np.where(mask_bool)[0]
    kp_f = [kps[i] for i in idx]
    if desc is None or len(idx) == 0:
        return kp_f, None
    des_f = desc[idx, :]
    return kp_f, des_f


def safe_bf_match(des1, des2):
    """
    BFMatcher for ORB (Hamming). Returns empty list if descriptors missing.
    """
    if des1 is None or des2 is None:
        return []
    if len(des1) == 0 or len(des2) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bf.match(des1, des2)


# ---------------- main experiment ----------------

def main(img1_path, img2_path, model_name="alike-t", device="cpu",
         nfeatures=5000, alike_topk=2000, match_radius=3.0, ransac_thresh=3.0):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None:
        raise SystemExit(f"Cannot read img1: {img1_path}")
    if img2 is None:
        raise SystemExit(f"Cannot read img2: {img2_path}")

    # load ALike model (как в demo_heatmap_mixed.py)
    cfg = configs[model_name].copy()
    cfg["device"] = device
    model = ALike(**cfg)
    model.eval()

    # ORB extraction on both images
    coords1, kp1, des1 = extract_orb_keypoints(img1, nfeatures=nfeatures)
    coords2, kp2, des2 = extract_orb_keypoints(img2, nfeatures=nfeatures)

    # ALike score map from img1 -> sparse ALike points (top-k maxima)
    inp1 = tensor_from_bgr(img1).to(model.device)
    dense1 = model.extract_dense_map(inp1, ret_dict=True)
    score_map1 = dense1["score_map"]  # (1,1,H,W)

    alike_pts1 = extract_alike_keypoints(score_map1, top_k=alike_topk)  # (M,2) float32

    # KDTree intersection: which ORB points are close to ALike points
    match_mask1 = match_kpts_kdtree(coords1, alike_pts1, radius=match_radius)

    # filtered ORB keypoints/descriptors for img1
    kp1_f, des1_f = filter_orb_by_mask(kp1, des1, match_mask1)

    # matching
    matches_orb = safe_bf_match(des1, des2)
    matches_filt = safe_bf_match(des1_f, des2)

    # reprojection errors
    err_orb = compute_mean_reproj_error(kp1, kp2, matches_orb, ransac_thresh=ransac_thresh)
    err_filt = compute_mean_reproj_error(kp1_f, kp2, matches_filt, ransac_thresh=ransac_thresh)

    print("\n=== Reprojection Error Experiment (KDTree ORB ∩ ALike) ===")
    print(f"Images          : {img1_path}  |  {img2_path}")
    print(f"ORB only        : {err_orb}")
    print(f"ORB ∩ ALike     : {err_filt}")
    print(f"Keypoints ORB   : {len(kp1)}")
    print(f"Keypoints filt  : {len(kp1_f)}")
    print(f"ALike top-k     : {len(alike_pts1)}")
    print(f"Match radius    : {match_radius} px")
    print(f"RANSAC thresh   : {ransac_thresh} px")
    print(f"Matches ORB     : {len(matches_orb)}")
    print(f"Matches filt    : {len(matches_filt)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean reprojection error: ORB vs ORB∩ALike (KDTree)")
    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    parser.add_argument("--model", choices=list(configs.keys()), default="alike-t")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nfeatures", type=int, default=5000)
    parser.add_argument("--alike-topk", type=int, default=2000)
    parser.add_argument("--match-radius", type=float, default=5.0)
    parser.add_argument("--ransac", type=float, default=3.0, help="RANSAC reprojection threshold in pixels")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(
        args.img1, args.img2,
        model_name=args.model,
        device=args.device,
        nfeatures=args.nfeatures,
        alike_topk=args.alike_topk,
        match_radius=args.match_radius,
        ransac_thresh=args.ransac,
    )
