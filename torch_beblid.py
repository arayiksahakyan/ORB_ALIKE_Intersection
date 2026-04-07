import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchBEBLID(nn.Module):
    def __init__(self, patch_size: int = 31, num_bits: int = 256):
        super().__init__()

        if num_bits != 256:
            raise ValueError("This version supports only 256-bit descriptors")

        self.patch_size = patch_size
        self.num_bits = num_bits

        pairs = self._make_pairs(num_bits, patch_size)
        self.register_buffer("pairs", pairs)

        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        self.register_buffer("bit_weights", weights)

    def _make_pairs(self, num_bits: int, patch_size: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(42)
        pts = torch.randint(0, patch_size, (num_bits, 4), generator=g)
        return pts.long()

    def extract_patches(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        n = keypoints.shape[1]
        ps = self.patch_size
        half = ps // 2

        x = keypoints[..., 0]
        y = keypoints[..., 1]

        xs = torch.linspace(-half, half, ps, device=image.device, dtype=image.dtype)
        ys = torch.linspace(-half, half, ps, device=image.device, dtype=image.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        gx = gx.view(1, 1, ps, ps)
        gy = gy.view(1, 1, ps, ps)

        x = x.view(b, n, 1, 1)
        y = y.view(b, n, 1, 1)

        sample_x = x + gx
        sample_y = y + gy

        if w > 1:
            sample_x = 2.0 * sample_x / (w - 1) - 1.0
        else:
            sample_x = torch.zeros_like(sample_x)

        if h > 1:
            sample_y = 2.0 * sample_y / (h - 1) - 1.0
        else:
            sample_y = torch.zeros_like(sample_y)

        grid = torch.stack([sample_x, sample_y], dim=-1)
        grid = grid.view(b * n, ps, ps, 2)

        img_rep = image.unsqueeze(1).repeat(1, n, 1, 1, 1).view(b * n, c, h, w)

        patches = F.grid_sample(
            img_rep,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        patches = patches.view(b, n, c, ps, ps)
        return patches

    def to_gray(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.shape[2] == 1:
            return patches[:, :, 0]

        r = patches[:, :, 0]
        g = patches[:, :, 1]
        b = patches[:, :, 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def forward(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(image, keypoints)
        gray = self.to_gray(patches)

        b, n, _, _ = gray.shape
        p = self.pairs

        y1, x1, y2, x2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]

        v1 = gray[:, :, y1, x1]
        v2 = gray[:, :, y2, x2]

        bits = (v1 > v2).to(torch.uint8)          # [B, N, 256]
        bits = bits.view(b, n, 32, 8)             # [B, N, 32, 8]

        weights = self.bit_weights.view(1, 1, 1, 8)
        packed = (bits * weights).sum(dim=-1).to(torch.uint8)   # [B, N, 32]

        return packed
