#!/usr/bin/env python3
import cv2
import os
import sys
import numpy as np


def extract_orb_keypoints(img_bgr, nfeatures=1000):
    """
    Extract ORB keypoints and descriptors from a BGR image.

    Returns:
        coords: np.ndarray of shape (N, 2) with (x, y) coordinates
        keypoints: list[cv2.KeyPoint]
        descriptors: np.ndarray or None
    """
    if img_bgr is None:
        raise ValueError("Input image/frame is None")

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img_bgr, None)
    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return coords, keypoints, descriptors


def run_on_image(path, nfeatures=2):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    coords, keypoints, desc = extract_orb_keypoints(img, nfeatures=nfeatures)
    print(f"[IMAGE] Detected {len(coords)} keypoints")
    if len(coords) > 0:
        print("First 10 keypoints (x, y):")
        print(coords[:10])

    img_kp = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("ORB keypoints (image)", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(path, nfeatures=2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    print("[VIDEO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break

        coords, keypoints, desc = extract_orb_keypoints(frame, nfeatures=nfeatures)

        frame_kp = cv2.drawKeypoints(
            frame,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        cv2.imshow("ORB keypoints (video)", frame_kp)

        # маленькая задержка и выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 orb_run.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        sys.exit(1)

    if is_image_file(path):
        run_on_image(path, nfeatures=500)
    else:
        # всё, что не картинка — считаем видео (mp4, avi, mkv, ...)
        run_on_video(path, nfeatures=3)


if __name__ == "__main__":
    main()
