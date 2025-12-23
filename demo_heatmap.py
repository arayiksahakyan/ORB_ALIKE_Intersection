import os
import cv2
import torch
import argparse
import logging
import numpy as np
from alike import ALike, configs
from orb_points import extract_orb_keypoints

def tensor_from_cv(img):
    # img: HxWx3 uint8 RGB or BGR (we assume BGR from cv2.imread)
    # convert to float tensor (1,3,H,W) in range 0..1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).to(torch.float32) / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t

def to_heatmap(score_map_tensor):
    # score_map_tensor: torch tensor (1,1,H,W)
    sm = score_map_tensor[0, 0].detach().cpu().numpy()  # H x W, float in model scale
    sm = np.clip(sm, 0.0, 1.0)
    heat = (sm * 255.0).astype('uint8')
    heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # HxW -> HxWx3 BGR
    return heatc

def run_on_image(model, filepath, window_name):
    img = cv2.imread(filepath)
    if img is None:
        raise IOError(f"Can't read image {filepath}")
    inp = tensor_from_cv(img).to(model.device)
    dense = model.extract_dense_map(inp, ret_dict=True)
    heat = to_heatmap(dense['score_map'])
    cv2.imshow(window_name, heat)
    cv2.waitKey(0)

def run_on_video(model, filepath, window_name):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise IOError(f"Can't open video {filepath}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = tensor_from_cv(frame).to(model.device)
        dense = model.extract_dense_map(inp, ret_dict=True)
        heat = to_heatmap(dense['score_map'])
        cv2.imshow(window_name, heat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def run_on_camera(model, cam_index, window_name):
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        raise IOError(f"Can't open camera {cam_index}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = tensor_from_cv(frame).to(model.device)
        dense = model.extract_dense_map(inp, ret_dict=True)
        heat = to_heatmap(dense['score_map'])
        
        # Now show heatmap 
        cv2.imshow("ALike heatmap", heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALike heatmap demo (heatmap only)')
    parser.add_argument('input', type=str, help='image path / video path / cameraX (e.g. camera0)')
    parser.add_argument('--model', choices=list(configs.keys()), default='alike-t')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda (we recommend cpu)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = configs[args.model].copy()
    cfg['device'] = args.device
    model = ALike(**cfg)

    window = 'ALike Heatmap'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    input_arg = args.input

    # decide mode
    if input_arg.startswith('camera'):
        run_on_camera(model, input_arg[6:], window)
    elif os.path.isfile(input_arg):
        # image or video? check extensions
        ext = os.path.splitext(input_arg)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.ppm']:
            run_on_image(model, input_arg, window)
        else:
            run_on_video(model, input_arg, window)
    else:
        raise IOError('Input must be path to file or "cameraX"')

    cv2.destroyAllWindows()

