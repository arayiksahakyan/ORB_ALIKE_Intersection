#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import logging
import numpy as np

from descriptor_match import (
    draw_matches_side_by_side,
    draw_tracks_on_current_frame,
    mutual_nearest_neighbors_hamming,
)
from onnx_pipeline import ALikeORB_BEBLID_ONNX


def diversify_scores_for_viz(
    keypoints: np.ndarray, scores: np.ndarray, min_dist_px: float
) -> np.ndarray:
    if min_dist_px <= 0:
        return scores
    xy = keypoints[0]
    s = scores[0]
    order = np.argsort(-s)
    keep = np.zeros(len(s), dtype=bool)
    picked = []
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
    if max_per_x_column > 0:
        return diversify_column_cap(keypoints, scores, max_per_x_column)
    if min_dist_px > 0:
        return diversify_scores_for_viz(keypoints, scores, min_dist_px)
    return scores


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested (%s) but torch.cuda.is_available() is False "
            "(CPU-only PyTorch, missing driver, etc.); using CPU.",
            name,
        )
        return torch.device("cpu")
    return dev


def tensor_from_bgr(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(device, non_blocking=device.type == "cuda")


def heatmap_to_bgr(heatmap: torch.Tensor) -> np.ndarray:
    hm = heatmap[0, 0].detach().cpu().numpy()
    hm = np.clip(hm, 0.0, 1.0)
    hm = (hm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(hm, cv2.COLORMAP_JET)


def draw_keypoints(img_bgr: np.ndarray, keypoints: torch.Tensor, scores: torch.Tensor = None) -> np.ndarray:
    vis = img_bgr.copy()

    kp = keypoints[0].detach().cpu().numpy()

    if scores is not None:
        sc = scores[0].detach().cpu().numpy()
    else:
        sc = None

    h, w = vis.shape[:2]

    for i, pt in enumerate(kp):
        if sc is not None and sc[i] <= 0:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))

        r = 3 if sc is not None else 2
        # Avoid clipped circles on edges (esp. y=0) merging into a horizontal green band.
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


def print_info(heatmap: torch.Tensor, keypoints: torch.Tensor, scores: torch.Tensor, descriptors: torch.Tensor):
    print("heatmap shape:", tuple(heatmap.shape))
    print("keypoints shape:", tuple(keypoints.shape))
    print("scores shape:", tuple(scores.shape))
    print("descriptors shape:", tuple(descriptors.shape))


def read_frame_from_video(path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0:
        cap.release()
        raise SystemExit(f"Frame index must be >= 0, got {frame_idx}")
    if frame_count > 0 and frame_idx >= frame_count:
        cap.release()
        raise SystemExit(
            f"Frame index {frame_idx} is out of range for video with {frame_count} frames"
        )

    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    if not ok:
        logging.warning("Could not seek directly to frame %d; trying read anyway", frame_idx)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise SystemExit(f"Cannot read frame {frame_idx} from video: {path}")
    return frame


def run_match_two_video_frames(
    model: ALikeORB_BEBLID_ONNX,
    path: str,
    frame_a: int,
    frame_b: int,
    device: torch.device,
    max_hamming: int,
) -> None:
    img_a = read_frame_from_video(path, frame_a)
    img_b = read_frame_from_video(path, frame_b)

    xa = tensor_from_bgr(img_a, device)
    xb = tensor_from_bgr(img_b, device)

    with torch.no_grad():
        _, kp_a, sc_a, desc_a = model(xa)
        _, kp_b, sc_b, desc_b = model(xb)

    kp_an = kp_a.detach().cpu().numpy()
    kp_bn = kp_b.detach().cpu().numpy()
    sc_an = sc_a.detach().cpu().numpy()
    sc_bn = sc_b.detach().cpu().numpy()
    desc_an = desc_a.detach().cpu().numpy()
    desc_bn = desc_b.detach().cpu().numpy()

    matches = mutual_nearest_neighbors_hamming(
        desc_an, desc_bn, sc_an, sc_bn, max_hamming=max_hamming
    )
    logging.info(
        "Video frame match: frame %d vs frame %d -> %d mutual matches (max Hamming=%d)",
        frame_a,
        frame_b,
        len(matches),
        max_hamming,
    )

    vis = draw_matches_side_by_side(img_a, img_b, kp_an, kp_bn, matches)
    win = f"Torch pipeline: frame {frame_a} vs frame {frame_b}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_match_two_images(
    model: ALikeORB_BEBLID_ONNX,
    path_a: str,
    path_b: str,
    device: torch.device,
    max_hamming: int,
) -> None:
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None:
        raise SystemExit(f"Cannot read image: {path_a}")
    if img_b is None:
        raise SystemExit(f"Cannot read image: {path_b}")

    xa = tensor_from_bgr(img_a, device)
    xb = tensor_from_bgr(img_b, device)

    with torch.no_grad():
        _, kp_a, sc_a, desc_a = model(xa)
        _, kp_b, sc_b, desc_b = model(xb)

    kp_an = kp_a.detach().cpu().numpy()
    kp_bn = kp_b.detach().cpu().numpy()
    sc_an = sc_a.detach().cpu().numpy()
    sc_bn = sc_b.detach().cpu().numpy()
    desc_an = desc_a.detach().cpu().numpy()
    desc_bn = desc_b.detach().cpu().numpy()

    matches = mutual_nearest_neighbors_hamming(
        desc_an, desc_bn, sc_an, sc_bn, max_hamming=max_hamming
    )
    logging.info(
        "Mutual nearest-neighbor matches: %d (max Hamming=%d)",
        len(matches),
        max_hamming,
    )

    vis = draw_matches_side_by_side(img_a, img_b, kp_an, kp_bn, matches)
    cv2.namedWindow("Torch pipeline: descriptor matches", cv2.WINDOW_NORMAL)
    cv2.imshow("Torch pipeline: descriptor matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_image(
    model: ALikeORB_BEBLID_ONNX,
    path: str,
    show_heatmap: bool,
    device: torch.device,
    viz_min_dist: float = 0.0,
    viz_max_per_x_column: int = 0,
):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    x = tensor_from_bgr(img, device)

    with torch.no_grad():
        heatmap, keypoints, scores, descriptors = model(x)

    print_info(heatmap, keypoints, scores, descriptors)

    kp_np = keypoints.detach().cpu().numpy()
    sc_np = scores.detach().cpu().numpy()
    sc_np = apply_viz_diversification(
        kp_np, sc_np, viz_min_dist, viz_max_per_x_column
    )
    sc_vis = torch.from_numpy(sc_np).to(device)
    vis_kp = draw_keypoints(img, keypoints, sc_vis)

    if show_heatmap:
        heatmap_bgr = heatmap_to_bgr(heatmap)
        vis = make_side_by_side(vis_kp, heatmap_bgr)
        window_name = "Torch pipeline: keypoints | heatmap"
    else:
        vis = vis_kp
        window_name = "Torch pipeline: keypoints"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_match_tracking(
    model: ALikeORB_BEBLID_ONNX,
    path: str,
    device: torch.device,
    max_hamming: int,
    output_video: str | None = None,
    save_frames: set[int] | None = None,
    frames_dir: str | None = None,
) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info(
        "Video tracking: prev↔curr mutual NN (Hamming). Cyan: track; orange: prev; green: curr. q=quit"
    )

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = None
    save_frames = save_frames or set()

    win = "Torch pipeline: video tracks (prev→curr)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, fw, fh)

    prev_kp = prev_sc = prev_desc = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        x = tensor_from_bgr(frame, device)
        with torch.no_grad():
            _hm, kp, sc, desc = model(x)

        kp_np = kp.detach().cpu().numpy()
        sc_np = sc.detach().cpu().numpy()
        desc_np = desc.detach().cpu().numpy()

        if prev_desc is not None:
            matches = mutual_nearest_neighbors_hamming(
                prev_desc, desc_np, prev_sc, sc_np, max_hamming=max_hamming
            )
            vis = draw_tracks_on_current_frame(frame, prev_kp, kp_np, matches)
            if frame_idx % 15 == 0:
                logging.info("frame %d: MNN matches %d", frame_idx, len(matches))
        else:
            vis = draw_keypoints(frame, kp, sc)

        if writer is None and output_video:
            h_vis, w_vis = vis.shape[:2]
            writer = make_video_writer(output_video, fps, (w_vis, h_vis))
            logging.info("Writing output video to %s", output_video)

        if writer is not None:
            writer.write(vis)

        save_frame_if_needed(frame_idx, vis, save_frames, frames_dir)

        prev_kp = np.copy(kp_np)
        prev_sc = np.copy(sc_np)
        prev_desc = np.copy(desc_np)

        cv2.imshow(win, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run_on_video(
    model: ALikeORB_BEBLID_ONNX,
    path: str,
    show_heatmap: bool,
    device: torch.device,
    viz_min_dist: float = 0.0,
    viz_max_per_x_column: int = 0,
    output_video: str | None = None,
    save_frames: set[int] | None = None,
    frames_dir: str | None = None,
):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info("Press 'q' to quit video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = None
    save_frames = save_frames or set()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if show_heatmap:
        window_w = w * 2
        window_h = h
        window_name = "Torch pipeline: keypoints | heatmap"
    else:
        window_w = w
        window_h = h
        window_name = "Torch pipeline: keypoints"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_w, window_h)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or cannot read frame")
            break

        x = tensor_from_bgr(frame, device)

        with torch.no_grad():
            heatmap, keypoints, scores, descriptors = model(x)

        if frame_idx % 10 == 0:
            print(f"\nframe {frame_idx}")
            print_info(heatmap, keypoints, scores, descriptors)

        kp_np = keypoints.detach().cpu().numpy()
        sc_np = scores.detach().cpu().numpy()
        sc_np = apply_viz_diversification(
            kp_np, sc_np, viz_min_dist, viz_max_per_x_column
        )
        sc_vis = torch.from_numpy(sc_np).to(device)
        vis_kp = draw_keypoints(frame, keypoints, sc_vis)

        if show_heatmap:
            heatmap_bgr = heatmap_to_bgr(heatmap)
            vis = make_side_by_side(vis_kp, heatmap_bgr)
        else:
            vis = vis_kp

        if writer is None and output_video:
            h_vis, w_vis = vis.shape[:2]
            writer = make_video_writer(output_video, fps, (w_vis, h_vis))
            logging.info("Writing output video to %s", output_video)

        if writer is not None:
            writer.write(vis)

        save_frame_if_needed(frame_idx, vis, save_frames, frames_dir)

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".webp"]


def build_model(args, device: torch.device):
    model = ALikeORB_BEBLID_ONNX(
        model_name=args.model,
        max_keypoints=args.max_keypoints,
        score_threshold=args.score_threshold,
        nms_kernel=args.nms_kernel,
        patch_size=args.patch_size,
        num_bits=args.num_bits,
        device=str(device),
    ).eval()

    return model


def parse_frame_list(text: str) -> set[int]:
    if not text.strip():
        return set()
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def parse_frame_pair(text: str) -> tuple[int, int] | None:
    if not text.strip():
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit("--match-video-frames must look like A,B for example 1,5")
    return int(parts[0]), int(parts[1])


def make_video_writer(path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, frame_size)


def save_frame_if_needed(
    frame_idx: int,
    frame: np.ndarray,
    frames_to_save: set[int],
    frames_dir: str | None,
) -> None:
    if frames_dir is None or frame_idx not in frames_to_save:
        return
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(frames_dir, f"frame_{frame_idx}.png")
    cv2.imwrite(out_path, frame)
    logging.info("Saved frame %d -> %s", frame_idx, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo for PyTorch pipeline: ALike -> ORB-like detector -> BEBLID-like descriptor"
    )
    parser.add_argument("input", type=str, help="path to image or video")
    parser.add_argument("--model", type=str, default="alike-t",
                        choices=["alike-t", "alike-s", "alike-n", "alike-l"])
    parser.add_argument("--max_keypoints", type=int, default=500)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--nms_kernel", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=31)
    parser.add_argument("--num_bits", type=int, default=256)
    parser.add_argument("--show_heatmap", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto: use CUDA if available, else CPU; or e.g. cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--viz-min-dist",
        type=float,
        default=0.0,
        metavar="PX",
        help="visualization: greedy min spacing (large PX leaves few points; see --viz-max-per-x-column)",
    )
    parser.add_argument(
        "--viz-max-per-x-column",
        type=int,
        default=0,
        metavar="N",
        help="visualization: max keypoints per integer x; overrides --viz-min-dist when >0",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        metavar="IMAGE",
        help="second image for descriptor matching (mutual NN, Hamming); main input must be an image",
    )
    parser.add_argument(
        "--match-video-frames",
        type=str,
        default="",
        metavar="A,B",
        help="for video input: show descriptor matches between two frame indices, e.g. 1,5",
    )
    parser.add_argument(
        "--max-hamming",
        type=int,
        default=80,
        help="max Hamming distance for a mutual NN pair (0–256)",
    )
    parser.add_argument(
        "--no-track-matches",
        action="store_true",
        help="video only: keypoints/heatmap only, no frame-to-frame descriptor matching (default video: tracks on)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="",
        help="save processed video to this file, e.g. output.mp4",
    )
    parser.add_argument(
        "--save-frames",
        type=str,
        default="",
        help="comma-separated frame indices to save, e.g. 1,5,20",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="saved_frames",
        help="directory where selected frames will be saved",
    )
    args = parser.parse_args()

    save_frames = parse_frame_list(args.save_frames)
    output_video = args.output_video if args.output_video else None
    match_video_frames = parse_frame_pair(args.match_video_frames)

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.input):
        raise SystemExit(f"Input path does not exist: {args.input}")

    device = resolve_device(args.device)
    if args.device == "auto":
        logging.info("Using device: %s (auto)", device)
    else:
        logging.info("Using device: %s", device)

    model = build_model(args, device)

    if args.match and match_video_frames is not None:
        raise SystemExit("Use either --match for two images or --match-video-frames for a video, not both")

    if args.match:
        if not is_image_file(args.input):
            raise SystemExit("--match requires the main input to be an image, not video")
        if not is_image_file(args.match):
            raise SystemExit("--match must be an image path")
        if not os.path.isfile(args.match):
            raise SystemExit(f"--match file not found: {args.match}")
        run_match_two_images(
            model,
            args.input,
            args.match,
            device,
            max_hamming=args.max_hamming,
        )
    elif match_video_frames is not None:
        if is_image_file(args.input):
            raise SystemExit("--match-video-frames requires the main input to be a video, not an image")
        frame_a, frame_b = match_video_frames
        run_match_two_video_frames(
            model,
            args.input,
            frame_a,
            frame_b,
            device,
            max_hamming=args.max_hamming,
        )
    elif is_image_file(args.input):
        run_on_image(
            model,
            args.input,
            args.show_heatmap,
            device,
            viz_min_dist=args.viz_min_dist,
            viz_max_per_x_column=args.viz_max_per_x_column,
        )
    elif args.no_track_matches:
        run_on_video(
            model,
            args.input,
            args.show_heatmap,
            device,
            viz_min_dist=args.viz_min_dist,
            viz_max_per_x_column=args.viz_max_per_x_column,
            output_video=output_video,
            save_frames=save_frames,
            frames_dir=args.frames_dir,
        )
    else:
        run_video_match_tracking(
            model,
            args.input,
            device,
            max_hamming=args.max_hamming,
            output_video=output_video,
            save_frames=save_frames,
            frames_dir=args.frames_dir,
        )
