import torch
from onnx_pipeline import ALikeORB_BEBLID_ONNX


def export_to_onnx(
    onnx_path="alike_orb_beblid.onnx",
    height=480,
    width=640
):
    model = ALikeORB_BEBLID_ONNX(
        model_name="alike-t",
        max_keypoints=500,
        score_threshold=0.05,
        nms_kernel=5,
        patch_size=31,
        num_bits=256
    ).eval()

    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32)

    export_kw = dict(
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["heatmap", "keypoints", "keypoint_scores", "descriptors"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "heatmap": {0: "batch", 2: "height", 3: "width"},
            "keypoints": {0: "batch", 1: "num_keypoints"},
            "keypoint_scores": {0: "batch", 1: "num_keypoints"},
            "descriptors": {0: "batch", 1: "num_keypoints"},
        },
    )
    # PyTorch 2.x defaults to dynamo exporter; this model exports reliably with the legacy path.
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path, **export_kw, dynamo=False
        )
    except TypeError:
        torch.onnx.export(model, dummy_input, onnx_path, **export_kw)

    print(f"Exported ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    export_to_onnx()
