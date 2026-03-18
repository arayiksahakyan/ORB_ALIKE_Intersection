import os
import cv2
import torch
import argparse
import numpy as np

from alike import ALike, configs


# =========================
# ALike utils
# =========================

def tensor_from_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t


def get_score_map(model: ALike, img_bgr: np.ndarray) -> np.ndarray:
    inp = tensor_from_bgr(img_bgr).to(model.device)
    with torch.no_grad():
        dense = model.extract_dense_map(inp, ret_dict=True)
    score_map = dense["score_map"][0, 0].detach().cpu().numpy()
    return score_map


# =========================
# BEBLID utils
# =========================

def create_beblid(scale_factor: float = 1.0, n_bits: int = 101):
    if not hasattr(cv2, "xfeatures2d"):
        raise RuntimeError("cv2.xfeatures2d not found. Install opencv-contrib-python.")
    if not hasattr(cv2.xfeatures2d, "BEBLID_create"):
        raise RuntimeError("cv2.xfeatures2d.BEBLID_create not found. Install opencv-contrib-python.")
    return cv2.xfeatures2d.BEBLID_create(scale_factor, n_bits)


# =========================
# ORB + filter + BEBLID
# =========================

def detect_orb_keypoints(img_bgr: np.ndarray, nfeatures: int = 1000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints = orb.detect(img_bgr, None)
    return keypoints


def filter_keypoints_by_heatmap(keypoints, score_np: np.ndarray, heatmap_th: float):
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
    if len(keypoints) == 0:
        return [], None

    beblid = create_beblid(scale_factor=scale_factor, n_bits=n_bits)
    keypoints, descriptors = beblid.compute(img_bgr, keypoints)
    return keypoints, descriptors


def extract_features(
    model: ALike,
    img_bgr: np.ndarray,
    nfeatures: int = 1000,
    heatmap_th: float = 0.05,
    beblid_scale_factor: float = 1.0,
    beblid_n_bits: int = 101
):
    score_np = get_score_map(model, img_bgr)

    orb_kps = detect_orb_keypoints(img_bgr, nfeatures=nfeatures)
    filt_kps = filter_keypoints_by_heatmap(orb_kps, score_np, heatmap_th)

    filt_kps, desc = compute_beblid_descriptors(
        img_bgr,
        filt_kps,
        scale_factor=beblid_scale_factor,
        n_bits=beblid_n_bits
    )

    return orb_kps, filt_kps, desc


# =========================
# Matching quality test
# =========================

def match_descriptors(desc1, desc2, ratio: float = 0.75):
    if desc1 is None or desc2 is None:
        return [], []

    if len(desc1) < 2 or len(desc2) < 2:
        return [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    return knn_matches, good


def compute_inliers_homography(kps1, kps2, matches, ransac_thresh: float = 3.0):
    if len(matches) < 4:
        return None, None, 0

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)

    if mask is None:
        return H, None, 0

    inliers = int(mask.sum())
    return H, mask, inliers


def draw_match_visualization(img1, kps1, img2, kps2, matches, max_matches=50, mask=None):
    matches_to_draw = matches[:max_matches]

    if mask is not None:
        mask_list = mask.ravel().tolist()[:len(matches_to_draw)]
        vis = cv2.drawMatches(
            img1, kps1,
            img2, kps2,
            matches_to_draw,
            None,
            matchesMask=mask_list,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    else:
        vis = cv2.drawMatches(
            img1, kps1,
            img2, kps2,
            matches_to_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    return vis


# =========================
# Frame loading
# =========================

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def load_two_video_frames(path: str, frame_a: int, frame_b: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_a >= total_frames or frame_b >= total_frames:
        cap.release()
        raise RuntimeError(f"Video has only {total_frames} frames, but requested {frame_a} and {frame_b}")

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read frame {idx}")
        return frame

    img1 = read_frame(frame_a)
    img2 = read_frame(frame_b)

    cap.release()
    return img1, img2


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Test descriptor quality with matching + RANSAC")
    parser.add_argument("input1", type=str, help="image1 path OR video path")
    parser.add_argument("input2", type=str, nargs="?", default=None, help="optional image2 path")
    parser.add_argument("--model", choices=list(configs.keys()), default="alike-t")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument("--heatmap_th", type=float, default=0.05)
    parser.add_argument("--beblid_scale_factor", type=float, default=1.0)
    parser.add_argument("--beblid_n_bits", type=int, default=101, choices=[100, 101, 102, 103])

    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test")
    parser.add_argument("--ransac_thresh", type=float, default=3.0)

    parser.add_argument("--frame_a", type=int, default=0, help="first frame if input is video")
    parser.add_argument("--frame_b", type=int, default=5, help="second frame if input is video")

    parser.add_argument("--show", action="store_true", help="show match visualization")
    parser.add_argument("--save", type=str, default="", help="save visualization image path")

    args = parser.parse_args()

    cfg = configs[args.model].copy()
    cfg["device"] = args.device
    model = ALike(**cfg)
    model.eval()

    # Case 1: two images
    if args.input2 is not None:
        img1 = load_image(args.input1)
        img2 = load_image(args.input2)

    # Case 2: one image only -> invalid
    elif is_image_file(args.input1):
        raise RuntimeError("For image mode provide two image paths: input1 input2")

    # Case 3: video
    else:
        img1, img2 = load_two_video_frames(args.input1, args.frame_a, args.frame_b)

    orb1, kps1, desc1 = extract_features(
        model, img1,
        nfeatures=args.nfeatures,
        heatmap_th=args.heatmap_th,
        beblid_scale_factor=args.beblid_scale_factor,
        beblid_n_bits=args.beblid_n_bits
    )

    orb2, kps2, desc2 = extract_features(
        model, img2,
        nfeatures=args.nfeatures,
        heatmap_th=args.heatmap_th,
        beblid_scale_factor=args.beblid_scale_factor,
        beblid_n_bits=args.beblid_n_bits
    )

    knn_matches, good_matches = match_descriptors(desc1, desc2, ratio=args.ratio)
    H, mask, inliers = compute_inliers_homography(kps1, kps2, good_matches, ransac_thresh=args.ransac_thresh)

    desc1_count = 0 if desc1 is None else len(desc1)
    desc2_count = 0 if desc2 is None else len(desc2)
    total_knn = len(knn_matches)
    total_good = len(good_matches)

    inlier_ratio = (inliers / total_good) if total_good > 0 else 0.0

    print("========== FEATURE STATS ==========")
    print(f"Image1 ORB keypoints total:        {len(orb1)}")
    print(f"Image1 filtered keypoints:         {len(kps1)}")
    print(f"Image1 descriptors:                {desc1_count}")
    print()
    print(f"Image2 ORB keypoints total:        {len(orb2)}")
    print(f"Image2 filtered keypoints:         {len(kps2)}")
    print(f"Image2 descriptors:                {desc2_count}")
    print()
    print("========== MATCHING STATS ==========")
    print(f"KNN matches:                       {total_knn}")
    print(f"Good matches after ratio test:     {total_good}")
    print(f"Inliers after RANSAC:              {inliers}")
    print(f"Inlier ratio:                      {inlier_ratio:.4f}")

    if total_good > 0:
        dists = [m.distance for m in good_matches]
        print(f"Good match distance min:           {min(dists):.2f}")
        print(f"Good match distance mean:          {sum(dists)/len(dists):.2f}")
        print(f"Good match distance max:           {max(dists):.2f}")

    if total_good < 10:
        print("\n[WARN] Very few good matches. Descriptor test may be unreliable.")

    if args.show or args.save:
        vis = draw_match_visualization(
            img1, kps1, img2, kps2,
            good_matches,
            max_matches=50,
            mask=mask
        )

        if args.save:
            cv2.imwrite(args.save, vis)
            print(f"\nSaved visualization to: {args.save}")

        if args.show:
            cv2.imshow("Good matches (inliers if RANSAC available)", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()