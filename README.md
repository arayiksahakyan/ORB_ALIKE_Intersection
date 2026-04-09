# ORB–ALike intersection

PyTorch pipeline that combines:

1. **ALike** — learned score heatmap (and descriptor map inside the backbone).
2. **Torch ORB detector** — Harris corner response fused with the heatmap (`harris × heatmap`), non-maximum suppression, then top‑K keypoints.
3. **BEBLID-style binary descriptors** — fixed random pixel pairs on grayscale patches, packed to bytes.

Use this repo to experiment with learned interest regions plus classical corner response, and to export subsets or the full stack to ONNX.

## Requirements

Python 3 with PyTorch. Install dependencies:

```bash
pip install -r requirements.txt
```

Pinned versions in `requirements.txt` are conservative; newer stacks may work but are not guaranteed.

## Model weights

Configs in `alike.py` expect checkpoints under `models/`:

- `models/alike-t.pth`, `alike-s.pth`, `alike-n.pth`, `alike-l.pth`

Place the official ALike weights (or your own compatible checkpoints) there. Without them, networks load with random weights and outputs are meaningless.

## Main pipeline (PyTorch)

Run on an image or video; press **q** to quit video mode.

```bash
python demo_torch_pipeline.py path/to/image.jpg --model alike-t --show_heatmap
python demo_torch_pipeline.py path/to/video.mp4 --max_keypoints 500
```

Options include `--model` (`alike-t` … `alike-l`), `--max_keypoints`, `--score_threshold`, `--nms_kernel`, `--patch_size`, `--num_bits` (256 only in the current BEBLID module).

## ONNX export

- **Heatmap only (ALike):** `export_onnx.py` → default `alike_heatmap.onnx`, opset 12, dynamic batch and spatial size.
- **Full pipeline:** `export_onnx_full.py` → default `alike_orb_beblid.onnx`, opset 16, outputs `heatmap`, `keypoints`, `keypoint_scores`, `descriptors`.

```bash
python export_onnx.py
python export_onnx_full.py
```

## Other scripts

| Script | Role |
|--------|------|
| `orb_run.py` | OpenCV ORB baseline on one image (visualize keypoints). |
| `demo.py` | ALike-focused demo (camera / video / image folder; see argparse in file). |
| `onnx_pipeline.py` | `ALikeORB_BEBLID_ONNX` module used by demos and export. |
| `hseq/extract.py`, `hseq/eval.py` | HPatches-style extraction / evaluation scaffolding (expects dataset paths as configured in those files). |

Heatmap / descriptor analysis utilities: `demo_heatmap.py`, `graph_heatmap.py`, `reproj_error_heatmap.py`, etc.

## Layout

- `alnet.py` / `alike.py` — backbone and ALike wrapper.
- `torch_orb.py` — differentiable Harris fusion and keypoint selection.
- `torch_beblid.py` — binary descriptors from sampled patches.
- `soft_detect.py` — soft detection helpers used by classic ALike-style extraction.

## License / attribution

ALike and related ideas follow their original licenses; cite the ALike paper and this project’s use case when publishing results.
