import cv2
import numpy as np
import argparse
import torch

from alike import ALike, configs
from demo_heatmap_mixed import (
    extract_orb_keypoints,
    tensor_from_bgr
)

# ---------------- geometry utils ----------------

def compute_mean_reproj_error(kp1, kp2, matches):
    if len(matches) < 4:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None

    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    proj = (H @ pts1_h.T).T
    proj = proj[:, :2] / proj[:, 2:]

    err = np.linalg.norm(proj - pts2, axis=1)
    return err[mask.ravel() == 1].mean()


# ---------------- ALike heatmap ----------------

def get_alike_heatmap(model, img_bgr):
    img_t = tensor_from_bgr(img_bgr)
    dense = model.extract_dense_map(img_t, ret_dict=True)
    score_map = dense["score_map"]
    return score_map[0, 0].detach().cpu().numpy()


# ---------------- ORB filtering ----------------

def filter_orb_by_heatmap(kps, desc, coords, heatmap, th):
    H, W = heatmap.shape
    kp_f, des_f = [], []

    for i, (x, y) in enumerate(coords):
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H:
            if heatmap[yi, xi] >= th:
                kp_f.append(kps[i])
                des_f.append(desc[i])

    if len(des_f) == 0:
        return [], None

    return kp_f, np.array(des_f)


# ---------------- main experiment ----------------

def main(img1_path, img2_path, model_name="alike-t", nfeatures=1000, th=0.002):

    # load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    assert img1 is not None and img2 is not None

    # load model
    cfg = configs[model_name].copy()
    cfg["device"] = "cpu"
    model = ALike(**cfg)
    model.eval()

    # ORB extraction
    coords1, kp1, des1 = extract_orb_keypoints(img1, nfeatures)
    coords2, kp2, des2 = extract_orb_keypoints(img2, nfeatures)

    # ALike heatmap (from first image)
    heatmap = get_alike_heatmap(model, img1)

    # filter ORB
    kp1_f, des1_f = filter_orb_by_heatmap(kp1, des1, coords1, heatmap, th)

    # matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = bf.match(des1, des2)
    matches_filt = bf.match(des1_f, des2)

    # reprojection error
    err_orb = compute_mean_reproj_error(kp1, kp2, matches_orb)
    err_filt = compute_mean_reproj_error(kp1_f, kp2, matches_filt)

    print("\n=== Reprojection Error Experiment ===")
    print(f"ORB only        : {err_orb}")
    print(f"ORB ∩ ALike     : {err_filt}")
    print(f"Keypoints ORB   : {len(kp1)}")
    print(f"Keypoints filt  : {len(kp1_f)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img1")
    parser.add_argument("img2")
    parser.add_argument("--nfeatures", type=int, default=5000)
    parser.add_argument("--th", type=float, default=0.05)
    args = parser.parse_args()

    main(args.img1, args.img2, nfeatures=args.nfeatures, th=args.th)
