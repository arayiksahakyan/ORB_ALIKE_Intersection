import logging
import os
import cv2
import torch
from copy import deepcopy
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import math

from alnet import ALNet
from soft_detect import DKD
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-l.pth')},
}


class ALike(ALNet):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5,
                 n_limit: int = 5000,
                 device: str = 'cpu',
                 model_path: str = ''
                 ):
        super().__init__(c1, c2, c3, c4, dim, single_head)
        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)

        # force CPU
        self.device = torch.device('cpu')

        if model_path != '':
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)
            # Remove self.to(self.device) → model is already on CPU
            # self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")
    def extract_dense_map(self, image, ret_dict=False):
        """
        Expects `image` as a torch tensor with shape (B, C, H, W).
        Returns score_map only, cropped to original HxW.
        """
        # make sure image is on the model device (we force CPU in __init__)
        image = image.to(self.device)

        b, c, h, w = image.shape
        # make sizes multiples of 32 (ALNet internal stride)
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w

        # pad if needed
        if h_ != h or w_ != w:
            img_pad = torch.zeros(b, c, h_, w_, device=self.device)
            img_pad[:, :, :h, :w] = image
            image = img_pad

        # forward through ALNet to get score map (ignore descriptor)
        with torch.no_grad():
            score_map, _ = super().forward(image)  # score_map shape: (B,1,H_,W_)

        # crop back to original size
        score_map = score_map[:, :, :h, :w]  # (B,1,H,W)

        if ret_dict:
            return {'score_map': score_map}

        return score_map


        def forward(self, image):
            """
            Forward pass for ONNX export.
            image: tensor (B,3,H,W)
            returns score_map: (B,1,H,W)
            """
            score_map, _ = super().forward(image)
            return score_map


if __name__ == '__main__':
    import numpy as np
    from thop import profile

    net = ALike(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)

    image = np.random.random((640, 480, 3)).astype(np.float32)
    flops, params = profile(net, inputs=(image, 9999, False), verbose=False)
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
