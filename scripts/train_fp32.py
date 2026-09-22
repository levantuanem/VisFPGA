"""
train_fp32.py
-------------
E1 for VisFPGA: train the M0 FP32 baseline (YOLOv8n on VisDrone2019-DET)
from scratch, self-contained — no dependency on any previous Colab
session's state. Run prepare_dataset.py first (or let this script trigger
the auto-download itself on first run).

Usage (fresh Colab GPU runtime):
    !pip install -q ultralytics
    !python scripts/prepare_dataset.py
    !python scripts/train_fp32.py \
        --model yolov8n.pt \
        --data VisDrone.yaml \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --out models/checkpoints/

Writes the run's best.pt to --out/best.pt (copied out of Ultralytics'
runs/detect/<name>/weights/ so downstream scripts have a stable path that
doesn't depend on the run name/number Ultralytics auto-assigns).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(
    model_name: str,
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    save_period: int,
    out_dir: str,
    run_name: str,
    resume: bool,
) -> Path:
    model = YOLO(model_name)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        save_period=save_period,
        name=run_name,
        resume=resume,
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but {best_pt} was not produced — check the run logs above.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / "best.pt"
    shutil.copy2(best_pt, dest)

    # Also copy final metrics summary if present, for the docs/results.md writeup
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, out_path / "train_results.csv")

    print(f"\nTraining complete.")
    print(f"  best.pt copied to: {dest}")
    print(f"  full run artifacts (curves, val images) at: {run_dir}")
    print(f"\nNext step: python scripts/export_onnx.py --weights {dest} --imgsz {imgsz}")
    return dest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the M0 FP32 baseline for VisFPGA (self-contained, fresh-session safe).")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model to fine-tune (yolov8n.pt, yolov8s.pt, ...)")
    p.add_argument("--data", type=str, default="VisDrone.yaml", help="Dataset yaml (Ultralytics resolves + auto-downloads if needed)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--save-period", type=int, default=10, help="Checkpoint save interval in epochs")
    p.add_argument("--out", type=str, default="models/checkpoints", dest="out_dir", help="Where to copy the final best.pt")
    p.add_argument("--name", type=str, default="m0_fp32_visdrone", dest="run_name", help="Ultralytics run name under runs/detect/")
    p.add_argument("--resume", action="store_true", help="Resume the named run if it was interrupted")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        save_period=args.save_period,
        out_dir=args.out_dir,
        run_name=args.run_name,
        resume=args.resume,
    )