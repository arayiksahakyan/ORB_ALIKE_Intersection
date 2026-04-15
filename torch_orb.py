import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class TorchORBDetector(nn.Module):
    def __init__(self, max_keypoints: int = 500, nms_kernel: int = 5, score_threshold: float = 0.05):
        super().__init__()
        self.max_keypoints = max_keypoints
        self.nms_kernel = nms_kernel
        self.score_threshold = score_threshold

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]], dtype=torch.float32
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]], dtype=torch.float32
        ).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def harris_response(self, gray: torch.Tensor) -> torch.Tensor:
        ix = F.conv2d(gray, self.sobel_x, padding=1)
        iy = F.conv2d(gray, self.sobel_y, padding=1)

        ixx = ix * ix
        iyy = iy * iy
        ixy = ix * iy

        sxx = F.avg_pool2d(ixx, kernel_size=3, stride=1, padding=1)
        syy = F.avg_pool2d(iyy, kernel_size=3, stride=1, padding=1)
        sxy = F.avg_pool2d(ixy, kernel_size=3, stride=1, padding=1)

        k = 0.04
        det = sxx * syy - sxy * sxy
        trace = sxx + syy
        return det - k * trace * trace

    def nms(self, score: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(score, kernel_size=self.nms_kernel, stride=1, padding=self.nms_kernel // 2)
        keep = (score == pooled).float()
        return score * keep

    def select_topk(self, score: torch.Tensor):
        b, _, h, w = score.shape
        flat = score.view(b, -1)

        vals, inds = torch.topk(flat, k=min(self.max_keypoints, flat.shape[1]), dim=1)
        xs = (inds % w).float()
        ys = (inds // w).float()

        coords = torch.stack([xs, ys], dim=-1)
        return coords, vals

    def forward(self, image: torch.Tensor, heatmap: torch.Tensor):
        gray = rgb_to_gray(image)

        orb_score = self.harris_response(gray)
        fused = orb_score * (0.5 + heatmap)
        fused = torch.where(fused >= self.score_threshold, fused, torch.zeros_like(fused))
        fused = self.nms(fused)

        keypoints, scores = self.select_topk(fused)
        return keypoints, scores, fused
