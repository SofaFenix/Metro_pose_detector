"""Example visualizations for temporal tuning."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T2F_ROOT = PROJECT_ROOT / "test2_with_filters"
SCRIPTS_DIR = T2F_ROOT / "scripts"
TEST2_META = PROJECT_ROOT / "test2" / "data" / "metadata_test2.json"

FIGURES_DIR = T2F_ROOT / "figures"
PREDICTIONS_PATH = T2F_ROOT / "results" / "predictions_C_optimized.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402

from comparison_core import (  # noqa: E402
    STATUS_ABNORMAL,
    STATUS_NORMAL,
    STATUS_UNRELIABLE,
    frame_file_path,
    infer_single_pose,
    load_image,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_VIDEOS = ("fall4.mp4", "normal2.mp4", "normal3.mp4")
CLIP_SEC_MAX = 10.0
CLIP_SEC_MIN = 5.0

COCO_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

STATUS_ORDER = [STATUS_NORMAL, STATUS_UNRELIABLE, STATUS_ABNORMAL]
STATUS_TO_Y = {s: i for i, s in enumerate(STATUS_ORDER)}
COLORS_BGR = {
    STATUS_NORMAL: (0, 220, 0),
    STATUS_ABNORMAL: (0, 0, 255),
    STATUS_UNRELIABLE: (0, 255, 255),
}

def load_fps_map() -> dict[str, float]:
    if not TEST2_META.is_file():
        return {}
    meta = json.loads(TEST2_META.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for key, entry in meta.items():
        fps = float(entry.get("fps") or 0)
        if fps > 0:
            out[key] = fps
            out[Path(key).stem.lower()] = fps
    return out

def plot_status_timeline(df_vid: pd.DataFrame, video_key: str, out_png: Path) -> None:
    df_vid = df_vid.sort_values("frame_index")
    t_ms = df_vid["timestamp_sec"].to_numpy(dtype=float) * 1000.0
    statuses = df_vid["final_status"].astype(str).tolist()
    y = np.array([STATUS_TO_Y.get(s, 0) for s in statuses], dtype=float)

    plt.figure(figsize=(11, 4))
    plt.step(t_ms, y, where="post", color="steelblue", linewidth=1.2)
    plt.scatter(t_ms, y, s=8, c="navy", alpha=0.35)
    plt.yticks(range(len(STATUS_ORDER)), STATUS_ORDER)
    plt.xlabel("Time (ms)")
    plt.ylabel("Status")
    plt.title(f"Temporal decision trace — {video_key}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def draw_overlay(
    bgr: np.ndarray,
    inf: dict[str, object],
    *,
    final_status: str,
    alert_triggered: bool,
) -> np.ndarray:
    out = bgr.copy()
    h, w_img = out.shape[:2]
    color = COLORS_BGR.get(final_status, (180, 180, 180))

    bbox = inf.get("bbox_xyxy")
    if isinstance(bbox, tuple) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    kp = inf.get("keypoints_xy")
    kc = inf.get("keypoints_conf")
    conf_t = float(config.CONFIDENCE_THRESHOLD)
    if isinstance(kp, np.ndarray) and isinstance(kc, np.ndarray) and kp.shape[0] >= 17:
        for a, b in COCO_EDGES:
            ia, ib = int(a), int(b)
            if ia < len(kc) and ib < len(kc) and float(kc[ia]) >= conf_t and float(kc[ib]) >= conf_t:
                pa = tuple(map(int, kp[ia]))
                pb = tuple(map(int, kp[ib]))
                cv2.line(out, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
        for idx in range(min(17, len(kp))):
            x, y = int(kp[idx][0]), int(kp[idx][1])
            if idx < len(kc) and float(kc[idx]) < conf_t:
                cv2.line(out, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(out, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(out, (x, y), 3, color, -1)

    lines = [
        f"status={final_status}",
        f"alert={alert_triggered}",
        f"vis={float(inf.get('visibility_rate', 0) or 0):.3f}",
    ]
    cv2.rectangle(out, (6, 6), (min(420, w_img - 6), 82), (20, 20, 20), -1)
    cv2.rectangle(out, (6, 6), (min(420, w_img - 6), 82), color, 2)
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)

    if alert_triggered:
        overlay = np.zeros_like(out)
        overlay[:, :] = (0, 0, 180)
        out = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)
    return out

def render_clip(
    model: YOLO,
    df_vid: pd.DataFrame,
    video_key: str,
    fps: float,
    out_mp4: Path,
) -> None:
    df_vid = df_vid.sort_values("frame_index").reset_index(drop=True)
    if df_vid.empty:
        logger.warning("Нет кадров для %s", video_key)
        return

    t_end = min(CLIP_SEC_MAX, float(df_vid["timestamp_sec"].max()) + 1e-3)
    t_end = max(t_end, min(CLIP_SEC_MIN, float(df_vid["timestamp_sec"].max()) + 1e-3))
    sub = df_vid.loc[df_vid["timestamp_sec"] <= t_end].copy()

    probe_path = frame_file_path(video_key, int(sub.iloc[0]["frame_index"]))
    probe = load_image(probe_path)
    if probe is None:
        logger.warning("Не удалось открыть первый кадр %s", probe_path)
        return
    height, width = probe.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Video {Path(video_key).stem}"):
        path = frame_file_path(video_key, int(row["frame_index"]))
        bgr = load_image(path)
        if bgr is None:
            continue
        try:
            inf = infer_single_pose(model, bgr)
        except Exception as exc:
            logger.warning("Inference failed %s: %s", path, exc)
            inf = {"ok": False, "visibility_rate": 0.0}
        drawn = draw_overlay(
            bgr,
            inf,
            final_status=str(row["final_status"]),
            alert_triggered=bool(row["alert_triggered"]),
        )
        if (drawn.shape[1], drawn.shape[0]) != (width, height):
            drawn = cv2.resize(drawn, (width, height))
        writer.write(drawn)

    writer.release()

def main() -> None:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(f"Отсутствует {PREDICTIONS_PATH}; зависимость: run_comparison.py.")

    pred = pd.read_csv(PREDICTIONS_PATH)
    fps_map = load_fps_map()
    model = YOLO(config.DEFAULT_MODEL, verbose=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for video_key in TARGET_VIDEOS:
        stem = Path(video_key).stem.lower()
        df_vid = pred.loc[pred["video_name"].astype(str).str.lower() == video_key.lower()].copy()
        if df_vid.empty:
            for alt in pred["video_name"].unique():
                if Path(str(alt)).stem.lower() == stem:
                    df_vid = pred.loc[pred["video_name"] == alt].copy()
                    video_key = str(alt)
                    break
        if df_vid.empty:
            logger.warning("Нет предсказаний для %s", video_key)
            continue

        fps = fps_map.get(video_key) or fps_map.get(stem) or 24.0
        if fps <= 0:
            fps = 24.0

        base = Path(video_key).stem
        png_path = FIGURES_DIR / f"examples_{base}_timeline.png"
        mp4_path = FIGURES_DIR / f"examples_{base}.mp4"

        try:
            plot_status_timeline(df_vid, video_key, png_path)
            render_clip(model, df_vid, video_key, fps, mp4_path)
            print(f"[OK] {base}: {png_path.name}, {mp4_path.name}")
        except Exception as exc:
            logger.warning("Сбой обработки %s: %s", video_key, exc)

if __name__ == "__main__":
    main()
