"""
Mutual nearest-neighbor matching for BEBLID-style binary descriptors (256 bits → 32 bytes).
Handles float [0,1] unpacked bits from ONNX or uint8 packed from PyTorch.
"""
from __future__ import annotations

import cv2
import numpy as np


def descriptors_to_packed_uint8(desc: np.ndarray) -> np.ndarray:
    """
    desc: (1, N, D) or (N, D), float or uint8.
    D=32: assumed packed bytes; D=256: unpacked bits (float threshold 0.5).
    Returns (N, 32) uint8.
    """
    d = np.asarray(desc)
    if d.ndim == 3:
        d = d[0]
    if d.shape[-1] == 32:
        if d.dtype in (np.float32, np.float64):
            return np.round(d).clip(0, 255).astype(np.uint8)
        return d.astype(np.uint8, copy=False)
    if d.shape[-1] == 256:
        if d.dtype in (np.float32, np.float64):
            bits = d > 0.5
        else:
            bits = d > 0
        bits_u8 = bits.astype(np.uint8)
        return np.packbits(bits_u8.reshape(-1, 256), axis=-1)
    raise ValueError(f"Expected descriptor last dim 32 or 256, got {d.shape}")


def hamming_distance_matrix_packed(a32: np.ndarray, b32: np.ndarray) -> np.ndarray:
    """a32, b32: (Na, 32) and (Nb, 32) uint8. Returns (Na, Nb) Hamming distances."""
    x = a32[:, None, :].astype(np.uint32) ^ b32[None, :, :].astype(np.uint32)
    c = np.zeros(x.shape, dtype=np.int32)
    for s in range(8):
        c += (x >> s) & 1
    return c.sum(axis=-1)


def valid_keypoint_mask(scores: np.ndarray) -> np.ndarray:
    """scores (1, N) — True where score > 0 (drops viz-zeroed slots)."""
    s = scores[0] if scores.ndim == 2 else scores
    return s > 0


def mutual_nearest_neighbors_hamming(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    max_hamming: int = 80,
) -> list[tuple[int, int]]:
    """
    Mutual nearest neighbors on Hamming distance, only between valid (score>0) keypoints.
    Returns list of (index_in_a, index_in_b) into the full N vectors.
    """
    pa = descriptors_to_packed_uint8(desc_a)
    pb = descriptors_to_packed_uint8(desc_b)
    ma = valid_keypoint_mask(scores_a)
    mb = valid_keypoint_mask(scores_b)
    ia = np.flatnonzero(ma)
    ib = np.flatnonzero(mb)
    if len(ia) == 0 or len(ib) == 0:
        return []

    sub_a = pa[ia]
    sub_b = pb[ib]
    dist = hamming_distance_matrix_packed(sub_a, sub_b)

    nn_ab = dist.argmin(axis=1)
    nn_ba = dist.argmin(axis=0)

    matches: list[tuple[int, int]] = []
    for i_sub, i_full in enumerate(ia):
        j_sub = int(nn_ab[i_sub])
        if dist[i_sub, j_sub] > max_hamming:
            continue
        if int(nn_ba[j_sub]) != i_sub:
            continue
        matches.append((int(i_full), int(ib[j_sub])))
    return matches


def draw_matches_side_by_side(
    img_a: np.ndarray,
    img_b: np.ndarray,
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    matches: list[tuple[int, int]],
    line_color: tuple[int, int, int] = (0, 255, 255),
    pt_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """keypoints (1, N, 2) in image coordinates. Matches are indices into N."""
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h = max(ha, hb)
    out = np.zeros((h, wa + wb, 3), dtype=np.uint8)
    out[:ha, :wa] = img_a
    out[:hb, wa : wa + wb] = img_b

    kpa = keypoints_a[0]
    kpb = keypoints_b[0]

    for ia, ib in matches:
        pa = (int(round(kpa[ia, 0])), int(round(kpa[ia, 1])))
        pb = (int(wa + round(kpb[ib, 0])), int(round(kpb[ib, 1])))
        cv2.line(out, pa, pb, line_color, 1, cv2.LINE_AA)
        cv2.circle(out, pa, 2, pt_color, -1, cv2.LINE_AA)
        cv2.circle(out, pb, 2, pt_color, -1, cv2.LINE_AA)
    return out


def draw_tracks_on_current_frame(
    img_bgr: np.ndarray,
    kp_prev: np.ndarray,
    kp_curr: np.ndarray,
    matches: list[tuple[int, int]],
    line_color: tuple[int, int, int] = (0, 255, 255),
    prev_pt_color: tuple[int, int, int] = (255, 128, 0),
    curr_pt_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw displacement vectors on the current frame: line from previous-frame position to
    current-frame position for each match (same pixel grid, fixed resolution video).
    kp_prev, kp_curr: (1, N, 2).
    """
    vis = img_bgr.copy()
    kpp = kp_prev[0]
    kpc = kp_curr[0]
    for ia, ib in matches:
        p0 = (int(round(kpp[ia, 0])), int(round(kpp[ia, 1])))
        p1 = (int(round(kpc[ib, 0])), int(round(kpc[ib, 1])))
        cv2.line(vis, p0, p1, line_color, 1, cv2.LINE_AA)
        cv2.circle(vis, p0, 2, prev_pt_color, -1, cv2.LINE_AA)
        cv2.circle(vis, p1, 2, curr_pt_color, -1, cv2.LINE_AA)
    return vis
