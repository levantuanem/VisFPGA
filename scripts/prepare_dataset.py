"""
prepare_dataset.py
-------------------
E0 utility for VisFPGA: download VisDrone2019-DET (via Ultralytics'
auto-download) and verify the folder structure, so a fresh Colab session
can rebuild the whole data pipeline with a single command — no manual
downloading, no dependency on a previous session's state.

Usage (fresh Colab runtime):
    !pip install -q ultralytics
    !python scripts/prepare_dataset.py --data-root /content/datasets

This writes/confirms:
    <data-root>/VisDrone/
        VisDrone2019-DET-train/
        VisDrone2019-DET-val/
        VisDrone2019-DET-test-dev/
    and prints the exact --data yaml path to pass to train_fp32.py.
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR


EXPECTED_SPLITS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
]


def prepare_visdrone(data_root: str | None) -> Path:
    # Ultralytics reads its own YOLODataset yaml (VisDrone.yaml ships with the
    # package) and auto-downloads on first use — triggering that here avoids
    # writing custom download/unzip logic.
    if data_root:
        # Override Ultralytics' default dataset directory for this run.
        import ultralytics.utils as u
        u.DATASETS_DIR = Path(data_root)
        dataset_dir = Path(data_root)
    else:
        dataset_dir = Path(DATASETS_DIR)

    print(f"Dataset root: {dataset_dir}")
    print("Triggering VisDrone auto-download via a 0-epoch dry run (downloads on first access)...")

    # A tiny, throwaway model load + val call is enough to force the
    # dataset check/download path without training anything.
    model = YOLO("yolov8n.pt")
    try:
        model.val(data="VisDrone.yaml", imgsz=320, batch=1, verbose=False, split="val")
    except Exception as e:  # noqa: BLE001 - surface but continue to verification
        print(f"Note: initial val() call raised {e!r} — checking downloaded files anyway.")

    visdrone_dir = dataset_dir / "VisDrone"
    missing = [s for s in EXPECTED_SPLITS if not (visdrone_dir / s).exists()]
    if missing:
        print(f"WARNING: missing expected splits: {missing}", file=sys.stderr)
        print("Re-run this script, or check disk space / network access on this runtime.", file=sys.stderr)
        sys.exit(1)

    for split in EXPECTED_SPLITS:
        n_images = len(list((visdrone_dir / split / "images").glob("*"))) if (visdrone_dir / split / "images").exists() else 0
        print(f"  {split}: {n_images} images found")

    print(f"\nDataset ready at: {visdrone_dir}")
    print("Pass --data VisDrone.yaml to train_fp32.py (Ultralytics resolves it from its dataset cache).")
    return visdrone_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and verify VisDrone2019-DET for VisFPGA.")
    p.add_argument("--data-root", type=str, default=None, help="Override Ultralytics dataset directory (default: its own cache dir, e.g. /content/datasets on Colab)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_visdrone(args.data_root)