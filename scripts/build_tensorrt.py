"""
build_tensorrt.py
------------------
E2 for VisFPGA: Post-Training Quantization (PTQ) baseline using TensorRT INT8.

Requires an NVIDIA GPU + TensorRT (e.g. Colab GPU runtime: `pip install
tensorrt` or use the TensorRT that ships with the Colab CUDA image).
Does NOT need Docker or a Linux server — only export_onnx.py's output and a
GPU are required, so this can run in the same Colab notebook used for
training the M0 baseline.

This uses Ultralytics' built-in TensorRT export (int8=True), which handles
calibration internally using a sample of the training/calibration set —
simpler and more robust than hand-rolling the TensorRT Python API for a
YOLO graph, and fine for establishing the E2 baseline the proposal expects
(PTQ mAP drop 6-10 pts vs M0, speedup >=2x).

Usage (Colab):
    !pip install tensorrt
    !python scripts/build_tensorrt.py \
        --weights models/checkpoints/best.pt \
        --data configs/dataset.yaml \
        --imgsz 640 \
        --out models/tensorrt/yolov8n_visdrone_int8.engine

Outputs:
    - the .engine file at --out
    - results/tables/e2_ptq_vs_fp32.csv comparing FP32 vs INT8 mAP50/mAP50-95/speed
"""

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO


def build_int8_engine(weights: str, data: str, imgsz: int, out: str, batch: int) -> Path:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    exported = model.export(
        format="engine",
        int8=True,
        data=data,           # calibration images are drawn from this dataset's train split
        imgsz=imgsz,
        batch=batch,
        workspace=4,          # GB, lower if the Colab GPU runs out of memory
    )
    exported_path = Path(exported)
    if exported_path.resolve() != out_path.resolve():
        out_path.write_bytes(exported_path.read_bytes())
        print(f"Copied {exported_path} -> {out_path}")

    return out_path


def benchmark(weights_fp32: str, engine_int8: str, data: str, imgsz: int, csv_out: str) -> None:
    rows = []

    for label, source in [("M0_FP32", weights_fp32), ("M1_PTQ_INT8", engine_int8)]:
        model = YOLO(source)

        t0 = time.perf_counter()
        metrics = model.val(data=data, imgsz=imgsz, split="val")
        elapsed = time.perf_counter() - t0

        rows.append({
            "config": label,
            "source": source,
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "val_wall_time_s": round(elapsed, 2),
        })
        print(f"{label}: mAP50={rows[-1]['mAP50']}  mAP50-95={rows[-1]['mAP50-95']}  val_time={rows[-1]['val_wall_time_s']}s")

    csv_path = Path(csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison table written to {csv_path}")

    fp32_map, int8_map = rows[0]["mAP50"], rows[1]["mAP50"]
    drop = fp32_map - int8_map
    print(f"\nmAP50 drop from PTQ: {drop:.4f} ({drop / fp32_map * 100:.1f}% relative)")
    print("Proposal target for E2: mAP drop of ~6-10 points vs M0 (in mAP50-95 terms) with >=2x speedup.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a TensorRT INT8 engine (PTQ, E2) and benchmark vs FP32.")
    p.add_argument("--weights", type=str, required=True, help="FP32 checkpoint, e.g. models/checkpoints/best.pt")
    p.add_argument("--data", type=str, required=True, help="Dataset yaml (e.g. configs/dataset.yaml or VisDrone.yaml)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="Calibration batch size")
    p.add_argument("--out", type=str, default="models/tensorrt/model_int8.engine")
    p.add_argument("--csv-out", type=str, default="results/tables/e2_ptq_vs_fp32.csv")
    p.add_argument("--skip-benchmark", action="store_true", help="Only build the engine, skip the mAP/speed comparison")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine_path = build_int8_engine(args.weights, args.data, args.imgsz, args.out, args.batch)
    if not args.skip_benchmark:
        benchmark(args.weights, str(engine_path), args.data, args.imgsz, args.csv_out)