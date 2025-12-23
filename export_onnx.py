import torch
from alike import ALike, configs

def export_to_onnx(model_name="alike-t", onnx_path="alike_heatmap.onnx"):
    cfg = configs[model_name].copy()
    cfg['device'] = 'cpu'   # Force CPU for export
    model = ALike(**cfg)
    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, 480, 640)  # (B, C, H, W)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,        # works well for conv/upsample
        do_constant_folding=True,
        input_names=['input'],
        output_names=['score_map'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'score_map': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )

    print(f"Exported ONNX model saved to: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()

