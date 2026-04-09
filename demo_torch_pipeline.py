#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import logging
import numpy as np

from onnx_pipeline import ALikeORB_BEBLID_ONNX


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
        x, y = int(round(pt[0])), int(round(pt[1]))

        if 0 <= x < w and 0 <= y < h:
            if sc is not None:
                r = 2 if sc[i] <= 0 else 3
            else:
                r = 2
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


def run_on_image(model: ALikeORB_BEBLID_ONNX, path: str, show_heatmap: bool, device: torch.device):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    x = tensor_from_bgr(img, device)

    with torch.no_grad():
        heatmap, keypoints, scores, descriptors = model(x)

    print_info(heatmap, keypoints, scores, descriptors)

    vis_kp = draw_keypoints(img, keypoints, scores)

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


def run_on_video(model: ALikeORB_BEBLID_ONNX, path: str, show_heatmap: bool, device: torch.device):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    logging.info("Press 'q' to quit video")

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

        vis_kp = draw_keypoints(frame, keypoints, scores)

        if show_heatmap:
            heatmap_bgr = heatmap_to_bgr(heatmap)
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.input):
        raise SystemExit(f"Input path does not exist: {args.input}")

    device = resolve_device(args.device)
    if args.device == "auto":
        logging.info("Using device: %s (auto)", device)
    else:
        logging.info("Using device: %s", device)

    model = build_model(args, device)

    if is_image_file(args.input):
        run_on_image(model, args.input, args.show_heatmap, device)
    else:
        run_on_video(model, args.input, args.show_heatmap, device)
