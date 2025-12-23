# orb_points.py
import cv2
import numpy as np
from typing import Tuple, List

def extract_orb_keypoints(
    img_bgr: np.ndarray,
    nfeatures: int = 1000
) -> Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray | None]:
    """
    Extract ORB keypoints and descriptors.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (as from cv2.imread or cv2.VideoCapture).
    nfeatures : int
        Maximum number of keypoints to retain.

    Returns
    -------
    coords : np.ndarray
        Array of shape (N, 2) with (x, y) pixel coordinates of keypoints.
        These coordinates are in the SAME coordinate system as the image
        (and your ALike heatmap, if you use the same frame).
    keypoints : list[cv2.KeyPoint]
        The original OpenCV keypoint objects.
    descriptors : np.ndarray or None
        ORB descriptors (N, 32) or None if no keypoints found.
    """
    if img_bgr is None:
        raise ValueError("Input image is None in extract_orb_keypoints")

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img_bgr, None)

    # (x, y) coordinates in pixel space, float -> convert to float32
    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # shape (N, 2)

    return coords, keypoints, descriptors


if __name__ == "__main__":
    # Small test: run ORB on an image and show result
    img_path = "testvd.mp4"
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    coords, kps, desc = extract_orb_keypoints(img, nfeatures=5000)
    print(f"Detected {len(coords)} keypoints")
    print("First 10 coordinates (x, y):")
    print(coords[:10])

    # Visualize
    img_kp = cv2.drawKeypoints(
        img, kps, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imshow("ORB keypoints", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
