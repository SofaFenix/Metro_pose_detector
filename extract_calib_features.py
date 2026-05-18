"""Extract theta, K_ar from calibration image sets."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import config

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

CONF_THRESHOLD = 0.5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_gt_class(label_path: Path) -> str:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return "UNRELIABLE"
    class_id = int(float(text.split()[0]))
    return "ABNORMAL" if class_id == 0 else "NORMAL" if class_id == 1 else "UNRELIABLE"

def compute_theta(keypoints_xy: np.ndarray) -> float:
    shoulders = (keypoints_xy[5] + keypoints_xy[6]) / 2.0
    hips = (keypoints_xy[11] + keypoints_xy[12]) / 2.0
    return abs(math.degrees(math.atan2((hips[0] - shoulders[0]), (hips[1] - shoulders[1]))))

def extract_for_dataset(dataset_name: str, output_csv: Path, histogram_path: Path, shuffled_csv: Path) -> pd.DataFrame:
    dataset_root = DATA_DIR / dataset_name
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    model = YOLO(config.DEFAULT_MODEL, verbose=False)
    rows: list[dict[str, float | str]] = []

    image_paths = sorted([path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS])
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        gt_class = parse_gt_class(label_path)
        if gt_class == "UNRELIABLE":
            continue

        result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            continue

        confs = result.boxes.conf.detach().cpu().numpy()
        valid_indices = np.where(confs > CONF_THRESHOLD)[0]
        if len(valid_indices) == 0:
            continue
        best_idx = valid_indices[np.argmax(confs[valid_indices])]

        xywh = result.boxes.xywh[best_idx].detach().cpu().numpy()
        w = float(xywh[2])
        h = float(xywh[3]) if float(xywh[3]) != 0 else np.nan
        k_ar = float(w / h) if not np.isnan(h) else np.nan

        keypoints_xy = result.keypoints.xy[best_idx].detach().cpu().numpy()
        keypoints_conf = result.keypoints.conf[best_idx].detach().cpu().numpy()
        theta = compute_theta(keypoints_xy)
        mean_conf = float(np.nanmean(keypoints_conf))

        rows.append(
            {
                "filename": image_path.name,
                "gt_class": gt_class,
                "theta": theta,
                "K_ar": k_ar,
                "mean_conf": mean_conf,
            }
        )

    df = pd.DataFrame(rows, columns=["filename", "gt_class", "theta", "K_ar", "mean_conf"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled_df.to_csv(shuffled_csv, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_histograms(df, histogram_path, dataset_name)
    print(f"{dataset_name}: saved {len(df)} rows to {output_csv}")
    return df

def plot_histograms(df: pd.DataFrame, output_path: Path, dataset_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cls, color in [("ABNORMAL", "red"), ("NORMAL", "blue")]:
        subset = df[df["gt_class"] == cls]
        if subset.empty:
            continue
        axes[0].hist(subset["theta"], bins=30, alpha=0.45, density=True, color=color, label=cls)
        axes[1].hist(subset["K_ar"], bins=30, alpha=0.45, density=True, color=color, label=cls)

    axes[0].set_title(f"Theta distribution ({dataset_name})")
    axes[0].set_xlabel("theta")
    axes[1].set_title(f"K_ar distribution ({dataset_name})")
    axes[1].set_xlabel("K_ar")
    for axis in axes:
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main() -> None:
    extract_for_dataset(
        dataset_name="dataset_A",
        output_csv=DATA_DIR / "features_predicted_A.csv",
        histogram_path=FIGURES_DIR / "calib_hist_A.png",
        shuffled_csv=DATA_DIR / "features_predicted_A_shuffled.csv",
    )
    extract_for_dataset(
        dataset_name="dataset_B",
        output_csv=DATA_DIR / "features_predicted_B.csv",
        histogram_path=FIGURES_DIR / "calib_hist_B.png",
        shuffled_csv=DATA_DIR / "features_predicted_B_shuffled.csv",
    )
    print("Feature extraction complete.")

if __name__ == "__main__":
    main()
