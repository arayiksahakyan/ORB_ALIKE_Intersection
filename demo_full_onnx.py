#!/usr/bin/env python3
"""
Run the full exported ONNX pipeline (export_onnx_full.py → alike_orb_beblid.onnx) via ONNX Runtime.

No PyTorch required at inference time. Input is resized to the export resolution (default 480×640)
so the graph matches what was traced; keypoints and heatmap are scaled back for display on the
original image size.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit(
        "Install onnxruntime: pip install onnxruntime  (or onnxruntime-gpu for CUDA EP)"
    ) from e

from descriptor_match import (
    draw_matches_side_by_side,
    draw_tracks_on_current_frame,
    mutual_nearest_neighbors_hamming,
)


def _pick_providers(prefer_gpu: bool) -> list[str]:
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def preprocess_bgr(
    img_bgr: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """RGB NCHW float32 [0,1] and (scale_x, scale_y) to map keypoints back to original image."""
    h0, w0 = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
    scale_x = w0 / float(width)
    scale_y = h0 / float(height)
    return x, (scale_x, scale_y)


def heatmap_to_bgr(heatmap: np.ndarray) -> np.ndarray:
    hm = heatmap[0, 0]
    hm = np.clip(hm, 0.0, 1.0)
    hm = (hm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(hm, cv2.COLORMAP_JET)


def diversify_scores_for_viz(
    keypoints: np.ndarray, scores: np.ndarray, min_dist_px: float
) -> np.ndarray:
    """Greedy keep by descending score with minimum Euclidean spacing (same pixel space as kp)."""
    if min_dist_px <= 0:
        return scores
    xy = keypoints[0]
    s = scores[0]
    order = np.argsort(-s)
    keep = np.zeros(len(s), dtype=bool)
    picked: list[tuple[float, float]] = []
    d2 = min_dist_px * min_dist_px
    for i in order:
        if s[i] <= 0:
            continue
        x, y = float(xy[i, 0]), float(xy[i, 1])
        bad = False
        for px, py in picked:
            dx, dy = x - px, y - py
            if dx * dx + dy * dy < d2:
                bad = True
                break
        if not bad:
            keep[i] = True
            picked.append((x, y))
    out = scores.copy()
    out[0, ~keep] = 0.0
    return out


def diversify_column_cap(
    keypoints: np.ndarray, scores: np.ndarray, max_per_x: int
) -> np.ndarray:
    """Keep at most max_per_x keypoints per integer x (highest scores first). Spreads dense vertical rails."""
    if max_per_x <= 0:
        return scores
    xy = keypoints[0]
    s = scores[0]
    order = np.argsort(-s)
    keep = np.zeros(len(s), dtype=bool)
    per_x: dict[int, int] = {}
    for i in order:
        if s[i] <= 0:
            continue
        xi = int(round(xy[i, 0]))
        c = per_x.get(xi, 0)
        if c >= max_per_x:
            continue
        keep[i] = True
        per_x[xi] = c + 1
    out = scores.copy()
    out[0, ~keep] = 0.0
    return out


def apply_viz_diversification(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_dist_px: float,
    max_per_x_column: int,
) -> np.ndarray:
    """Column cap (if N>0) overrides min-distance; both off leaves all points for drawing."""
    if max_per_x_column > 0:
        return diversify_column_cap(keypoints, scores, max_per_x_column)
    if min_dist_px > 0:
        return diversify_scores_for_viz(keypoints, scores, min_dist_px)
    return scores


def draw_keypoints(
    img_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    vis = img_bgr.copy()
    kp = keypoints[0]
    sc = scores[0] if scores is not None else None
    h, w = vis.shape[:2]

    for i, pt in enumerate(kp):
        if sc is not None and sc[i] <= 0:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        if sc is not None:
            r = 3
        else:
            r = 2
        # Skip if circle would clip at the border — clipped semicircles along y=0 (or x=0)
        # merge visually into a solid green "line", not individual dots.
        if r <= x < w - r and r <= y < h - r:
            cv2.circle(vis, (x, y), r, (0, 255, 0), -1)
    return vis


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    h = max(h1, h2)
    if h1 != h:
        scale = h / h1
        left = cv2.resize(left, (int(w1 * scale), h))
    if h2 != h:
        scale = h / h2
        right = cv2.resize(right, (int(w2 * scale), h))
    return np.hstack([left, right])


def print_info(
    heatmap: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    descriptors: np.ndarray,
) -> None:
    print("heatmap shape:", heatmap.shape)
    print("keypoints shape:", keypoints.shape)
    print("scores shape:", scores.shape)
    print("descriptors shape:", descriptors.shape)


def log_keypoint_x_debug(keypoints: np.ndarray, image_width: int, frame_idx: int) -> None:
    """If many keypoints share few x values, vertical 'lines' are data clustering, not drawing."""
    xs = np.asarray(keypoints[0, :, 0]).ravel()
    n = xs.size
    if n == 0:
        return
    xr = np.round(xs).astype(np.int32)
    uniq = np.unique(xr)
    logging.info(
        "frame %d: keypoint x: unique columns=%d / %d points (image width=%d)",
        frame_idx,
        len(uniq),
        n,
        image_width,
    )
    uniq_x, cnt = np.unique(xr, return_counts=True)
    order = np.argsort(-cnt)
    top_k = min(20, len(uniq_x))
    top_idx = order[:top_k]
    logging.info(
        "frame %d: top x columns (x, count): %s",
        frame_idx,
        list(zip(uniq_x[top_idx].tolist(), cnt[top_idx].tolist())),
    )
    # Normalized x histogram: spikes at 1/3, 1/2 suggest periodic bias
    u = np.clip(xs / float(max(image_width, 1)), 0.0, 1.0)
    hist, edges = np.histogram(u, bins=24, range=(0.0, 1.0))
    peak_bins = np.argsort(hist)[-5:][::-1]
    logging.info(
        "frame %d: x/W density peaks (bin start..end, count): %s",
        frame_idx,
        [(f"{edges[i]:.3f}-{edges[i+1]:.3f}", int(hist[i])) for i in peak_bins],
    )


def scale_outputs_to_original(
    keypoints: np.ndarray,
    scores: np.ndarray,
    heatmap: np.ndarray,
    scale_xy: tuple[float, float],
    orig_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy = scale_xy
    oh, ow = orig_hw
    kp = keypoints.copy()
    kp[..., 0] *= sx
    kp[..., 1] *= sy
    sc = scores.copy()
    hm = heatmap[0, 0]
    hm_up = cv2.resize(hm, (ow, oh), interpolation=cv2.INTER_LINEAR)
    heatmap_full = hm_up[np.newaxis, np.newaxis, ...]
    return kp, sc, heatmap_full


def run_session(
    sess: ort.InferenceSession, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hm, kp, sc, desc = sess.run(None, {"input": x})
    return hm, kp, sc, desc


def run_match_two_images(
    sess: ort.InferenceSession,
    path_a: str,
    path_b: str,
    height: int,
    width: int,
    max_hamming: int,
) -> None:
    """Run pipeline on two images and show mutual nearest-neighbor descriptor matches."""
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None:
        raise SystemExit(f"Cannot read image: {path_a}")
    if img_b is None:
        raise SystemExit(f"Cannot read image: {path_b}")

    xa, scale_a = preprocess_bgr(img_a, height, width)
    xb, scale_b = preprocess_bgr(img_b, height, width)

    hm_a, kp_a, sc_a, desc_a = run_session(sess, xa)
    hm_b, kp_b, sc_b, desc_b = run_session(sess, xb)

    kp_ad, sc_ad, _ = scale_outputs_to_original(
        kp_a, sc_a, hm_a, scale_a, img_a.shape[:2]
    )
    kp_bd, sc_bd, _ = scale_outputs_to_original(
        kp_b, sc_b, hm_b, scale_b, img_b.shape[:2]
    )

    matches = mutual_nearest_neighbors_hamming(
        desc_a, desc_b, sc_ad, sc_bd, max_hamming=max_hamming
    )
    logging.info(
        "Mutual nearest-neighbor matches: %d (max Hamming=%d)",
        len(matches),
        max_hamming,
    )

    vis = draw_matches_side_by_side(img_a, img_b, kp_ad, kp_bd, matches)
    cv2.namedWindow("ONNX full: descriptor matches", cv2.WINDOW_NORMAL)
    cv2.imshow("ONNX full: descriptor matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_image(
    sess: ort.InferenceSession,
    path: str,
    show_heatmap: bool,
    height: int,
    width: int,
    debug_kp_x: bool = False,
    viz_min_dist: float = 0.0,
    viz_max_per_x_column: int = 0,
) -> None:
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    x, scale = preprocess_bgr(img, height, width)
    hm, kp, sc, desc = run_session(sess, x)

    kp_d, sc_d, hm_d = scale_outputs_to_original(kp, sc, hm, scale, img.shape[:2])

    print_info(hm_d, kp_d, sc_d, desc)
    if debug_kp_x:
        log_keypoint_x_debug(kp_d, img.shape[1], 0)

    sc_vis = apply_viz_diversification(
        kp_d, sc_d, viz_min_dist, viz_max_per_x_column
    )
    vis_kp = draw_keypoints(img, kp_d, sc_vis)

    if show_heatmap:
        heatmap_bgr = heatmap_to_bgr(hm_d)
        vis = make_side_by_side(vis_kp, heatmap_bgr)
        window_name = "ONNX full: keypoints | heatmap"
    else:
        vis = vis_kp
        window_name = "ONNX full: keypoints"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_match_tracking(
    sess: ort.InferenceSession,
    path: str,
    height: int,
    width: int,
    max_hamming: int,
) -> None:
    """
    Consecutive frames: match descriptors t-1 vs t, draw lines from previous keypoint
    positions to current positions on the current frame.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info(
        "Video tracking: prev↔curr mutual NN (Hamming). Cyan: track; orange: prev; green: curr. q=quit"
    )

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    win = "ONNX full: video tracks (prev→curr)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, fw, fh)

    prev_kp = None
    prev_sc = None
    prev_desc = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        x, scale = preprocess_bgr(frame, height, width)
        hm, kp, sc, desc = run_session(sess, x)
        kp_d, sc_d, _hm_d = scale_outputs_to_original(
            kp, sc, hm, scale, frame.shape[:2]
        )

        if prev_desc is not None:
            matches = mutual_nearest_neighbors_hamming(
                prev_desc, desc, prev_sc, sc_d, max_hamming=max_hamming
            )
            vis = draw_tracks_on_current_frame(frame, prev_kp, kp_d, matches)
            if frame_idx % 15 == 0:
                logging.info(
                    "frame %d: MNN matches %d",
                    frame_idx,
                    len(matches),
                )
        else:
            vis = draw_keypoints(frame, kp_d, sc_d)

        prev_kp = np.copy(kp_d)
        prev_sc = np.copy(sc_d)
        prev_desc = np.copy(desc)

        cv2.imshow(win, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def run_on_video(
    sess: ort.InferenceSession,
    path: str,
    show_heatmap: bool,
    height: int,
    width: int,
    debug_kp_x: bool = False,
    viz_min_dist: float = 0.0,
    viz_max_per_x_column: int = 0,
) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info("Press 'q' to quit video")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if show_heatmap:
        window_w = fw * 2
        window_h = fh
        window_name = "ONNX full: keypoints | heatmap"
    else:
        window_w = fw
        window_h = fh
        window_name = "ONNX full: keypoints"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_w, window_h)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        x, scale = preprocess_bgr(frame, height, width)
        hm, kp, sc, desc = run_session(sess, x)

        kp_d, sc_d, hm_d = scale_outputs_to_original(
            kp, sc, hm, scale, frame.shape[:2]
        )

        if frame_idx % 10 == 0:
            print(f"\nframe {frame_idx}")
            print_info(hm_d, kp_d, sc_d, desc)
        if debug_kp_x and frame_idx % 30 == 0:
            log_keypoint_x_debug(kp_d, frame.shape[1], frame_idx)

        sc_vis = apply_viz_diversification(
            kp_d, sc_d, viz_min_dist, viz_max_per_x_column
        )
        vis_kp = draw_keypoints(frame, kp_d, sc_vis)

        if show_heatmap:
            heatmap_bgr = heatmap_to_bgr(hm_d)
            vis = make_side_by_side(vis_kp, heatmap_bgr)
        else:
            vis = vis_kp

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".ppm",
        ".webp",
    ]


def main() -> None:
    root = Path(__file__).resolve().parent
    default_onnx = root / "alike_orb_beblid.onnx"

    parser = argparse.ArgumentParser(
        description="Demo: full ONNX pipeline (ALike + ORB + BEBLID) via ONNX Runtime"
    )
    parser.add_argument("input", type=str, help="path to image or video")
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(default_onnx),
        help=f"path to full ONNX model (default: {default_onnx.name} next to this script)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="resize height fed to the network (must match export tracing; default 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="resize width fed to the network (default 640)",
    )
    parser.add_argument("--show_heatmap", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use CPU execution provider only (default: try CUDA EP if available)",
    )
    parser.add_argument(
        "--debug-kp-x",
        action="store_true",
        help="log keypoint x clustering per frame (video: every 30 frames) to verify vertical bands",
    )
    parser.add_argument(
        "--viz-min-dist",
        type=float,
        default=0.0,
        metavar="PX",
        help="visualization: greedy min spacing PX px (large values leave few dots; prefer --viz-max-per-x-column)",
    )
    parser.add_argument(
        "--viz-max-per-x-column",
        type=int,
        default=0,
        metavar="N",
        help="visualization: cap keypoints per integer x (e.g. 100 keeps up to 500 with 5 rails). Overrides --viz-min-dist when >0",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        metavar="IMAGE",
        help="second image for descriptor matching (mutual nearest neighbors, Hamming). Requires first input to be an image.",
    )
    parser.add_argument(
        "--max-hamming",
        type=int,
        default=80,
        help="max Hamming distance (0–256) for a mutual NN pair to be kept",
    )
    parser.add_argument(
        "--no-track-matches",
        action="store_true",
        help="video only: disable consecutive-frame matching; show keypoints/heatmap only (default for video is prev→curr tracks)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.input):
        raise SystemExit(f"Input path does not exist: {args.input}")
    if not os.path.isfile(args.onnx):
        raise SystemExit(
            f"ONNX model not found: {args.onnx}\n"
            "Generate it with: python export_onnx_full.py"
        )

    providers = _pick_providers(prefer_gpu=not args.cpu)
    logging.info("ONNX Runtime providers: %s", providers)

    sess = ort.InferenceSession(args.onnx, providers=providers)

    if args.match:
        if not is_image_file(args.input):
            raise SystemExit("--match requires the main input to be an image, not video")
        if not is_image_file(args.match):
            raise SystemExit("--match must be an image path")
        if not os.path.isfile(args.match):
            raise SystemExit(f"--match file not found: {args.match}")
        run_match_two_images(
            sess,
            args.input,
            args.match,
            args.height,
            args.width,
            max_hamming=args.max_hamming,
        )
    elif is_image_file(args.input):
        run_on_image(
            sess,
            args.input,
            args.show_heatmap,
            args.height,
            args.width,
            debug_kp_x=args.debug_kp_x,
            viz_min_dist=args.viz_min_dist,
            viz_max_per_x_column=args.viz_max_per_x_column,
        )
    elif args.no_track_matches:
        run_on_video(
            sess,
            args.input,
            args.show_heatmap,
            args.height,
            args.width,
            debug_kp_x=args.debug_kp_x,
            viz_min_dist=args.viz_min_dist,
            viz_max_per_x_column=args.viz_max_per_x_column,
        )
    else:
        run_video_match_tracking(
            sess,
            args.input,
            args.height,
            args.width,
            max_hamming=args.max_hamming,
        )


if __name__ == "__main__":
    main()
