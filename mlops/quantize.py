"""
SatQuery AI - Model Quantization & Acceleration
Owner: Chhavi (AI & Deep Learning Lead + MLOps)
"""

from pathlib import Path

import torch


def export_to_onnx_fp16(model, dummy_input, output_path: Path):
    """
    Export a PyTorch model to ONNX format.

    The model is switched to evaluation mode before export.
    The current implementation performs ONNX export and keeps
    the interface ready for future FP16 optimization.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    print(f"Exporting model to ONNX: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

    return output_path