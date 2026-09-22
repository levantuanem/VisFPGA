"""
export_onnx.py
---------------
E0 utility for VisFPGA: export a trained Ultralytics YOLO checkpoint (M0 FP32
baseline) to ONNX, ready for the PTQ (E2) and FINN/QONNX (E3-E4) stages.

Runs anywhere ultralytics + torch are installed — no GPU, TensorRT, or
Brevitas required. Safe to run today on Colab or on your own machine.

Usage:
    python scripts/export_onnx.py \
        --weights models/checkpoints/best.pt \
        --out models/onnx/yolov8n_visdrone_fp32.onnx \
        --imgsz 640

On Colab (after training), this is just:
    !python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_onnx(weights: str, out: str, imgsz: int, opset: int, dynamic: bool, simplify: bool) -> Path:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}. "
            "Point --weights at your trained best.pt (e.g. from the Colab run)."
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    # Ultralytics' own exporter writes the .onnx next to the source weights
    # by default, so we export then move it to the requested output path.
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    exported_path = Path(exported)

    if exported_path.resolve() != out_path.resolve():
        out_path.write_bytes(exported_path.read_bytes())
        print(f"Copied {exported_path} -> {out_path}")

    print(f"Done. ONNX model ready at: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a YOLO checkpoint to ONNX for VisFPGA.")
    p.add_argument("--weights", type=str, required=True, help="Path to trained .pt checkpoint (e.g. models/checkpoints/best.pt)")
    p.add_argument("--out", type=str, default="models/onnx/model_fp32.onnx", help="Output .onnx path")
    p.add_argument("--imgsz", type=int, default=640, help="Export image size (match training imgsz)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--dynamic", action="store_true", help="Export with dynamic input shapes (usually keep off for FPGA-bound flows)")
    p.add_argument("--simplify", action="store_true", default=True, help="Run onnx-simplifier on the exported graph (default: on)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        weights=args.weights,
        out=args.out,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )