import torch
import torch.nn as nn

from alike import ALike, configs
from torch_orb import TorchORBDetector
from torch_beblid import TorchBEBLID


class ALikeORB_BEBLID_ONNX(nn.Module):
    def __init__(
        self,
        model_name: str = "alike-t",
        max_keypoints: int = 500,
        score_threshold: float = 0.05,
        nms_kernel: int = 5,
        patch_size: int = 31,
        num_bits: int = 256
    ):
        super().__init__()

        cfg = configs[model_name].copy()
        cfg["device"] = "cpu"

        self.alike = ALike(**cfg).eval()
        self.orb = TorchORBDetector(
            max_keypoints=max_keypoints,
            nms_kernel=nms_kernel,
            score_threshold=score_threshold
        )
        self.beblid = TorchBEBLID(
            patch_size=patch_size,
            num_bits=num_bits
        )

    def forward(self, image: torch.Tensor):
        heatmap = self.alike(image)
        keypoints, kp_scores, fused_score = self.orb(image, heatmap)
        descriptors = self.beblid(image, keypoints)

        return heatmap, keypoints, kp_scores, descriptors
