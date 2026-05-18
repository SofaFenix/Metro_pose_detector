"""Archive pipeline: YOLO-pose, geometry, temporal filter, predictions CSV."""
from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

import config

PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVE_FRAMES_ROOT = PROJECT_ROOT / "data" / "archive_frames"
GT_CSV_PATH = PROJECT_ROOT / "data" / "archive_annotations" / "frame_level_gt.csv"
PREDICTIONS_CSV_PATH = PROJECT_ROOT / "data" / "predictions.csv"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def video_bucket(video_name: str) -> str:
    """Разделение на notfall / fall / прочее (прочее — без доп. поредения)."""
    stem = Path(video_name).stem.lower()
    if stem.startswith("notfall"):
        return "notfall"
    if stem.startswith("fall"):
        return "fall"
    return "other"

def downsample_gt_per_video(df: pd.DataFrame) -> pd.DataFrame:
    """notfall: ~1/4 кадров — берём каждый 4-й по порядку frame_index (0, 4, 8, …)."""
    chunks: list[pd.DataFrame] = []
    for name, grp in df.groupby("video_name", sort=False):
        grp = grp.sort_values("frame_index", kind="mergesort")
        bucket = video_bucket(str(name))

        if bucket == "notfall":
            sampled = grp.iloc[::4]
        elif bucket == "fall":
            n = len(grp)
            pos = np.arange(n, dtype=np.int64)
            mask = (pos % 3) != 2
            sampled = grp.iloc[mask]
        else:
            sampled = grp

        if sampled.empty:
            sampled = grp.head(1)
        chunks.append(sampled)

    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)

def load_image(image_path: Path) -> np.ndarray | None:
    """Чтение BGR изображения; fallback для Unicode (Windows)."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is not None and img.size > 0:
        return img
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            return None
        return decoded
    except Exception as exc:
        logger.warning("Не удалось прочитать кадр %s: %s", image_path, exc)
        return None

def frame_file_path(video_name: str, frame_index: int) -> Path:
    stem = Path(video_name).stem
    return ARCHIVE_FRAMES_ROOT / stem / f"frame_{frame_index:05d}.jpg"

def compute_theta_deg(keypoints_xy: np.ndarray) -> float:
    shoulders = (keypoints_xy[5] + keypoints_xy[6]) / 2.0
    hips = (keypoints_xy[11] + keypoints_xy[12]) / 2.0
    return abs(math.degrees(math.atan2((hips[0] - shoulders[0]), (hips[1] - shoulders[1]))))

def infer_single_pose(
    model: YOLO,
    frame_bgr: np.ndarray | None,
    conf_thresh: float = 0.5,
) -> dict[str, object]:
    """
    Одна детекция с max confidence. При битом кадре / пустом результате возвращает нули/NaN и флаги ошибки.
    """
    empty: dict[str, object] = {
        "ok": False,
        "theta": float("nan"),
        "k_ar": float("nan"),
        "visibility_rate": 0.0,
        "mean_conf": 0.0,
        "is_abnormal_raw": False,
        "bbox_xyxy": None,
        "keypoints_xy": None,
        "keypoints_conf": None,
    }
    if frame_bgr is None or frame_bgr.size == 0:
        return empty

    try:
        results = model.predict(frame_bgr, conf=float(conf_thresh), verbose=False)
    except Exception as exc:
        logger.warning("YOLO predict error: %s", exc)
        return empty

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return empty

    res = results[0]
    confs = res.boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    if confs[best_idx] <= conf_thresh:
        return empty

    xyxy = res.boxes.xyxy[best_idx].detach().cpu().numpy()
    x1, y1, x2, y2 = map(float, xyxy)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    k_ar = float(w / h)

    if res.keypoints is None or len(res.keypoints) <= best_idx:
        return {
            **empty,
            "k_ar": k_ar,
            "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
        }

    kpt_xy = res.keypoints.xy[best_idx].detach().cpu().numpy()
    kpt_conf = res.keypoints.conf[best_idx].detach().cpu().numpy()
    theta = compute_theta_deg(kpt_xy)
    visibility_rate = float(np.mean(kpt_conf >= config.CONFIDENCE_THRESHOLD))
    mean_conf = float(np.mean(kpt_conf))
    raw = (theta > float(config.THETA_THRESHOLD) and k_ar > float(config.K_BASE)) or (
        k_ar > float(config.K_CRITICAL)
    )

    return {
        "ok": True,
        "theta": float(theta),
        "k_ar": float(k_ar),
        "visibility_rate": visibility_rate,
        "mean_conf": mean_conf,
        "is_abnormal_raw": bool(raw),
        "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
        "keypoints_xy": kpt_xy.astype(np.float32),
        "keypoints_conf": kpt_conf.astype(np.float32),
    }

class TemporalStabilizer:
    """TEMPORAL: окно deque + алерт MIN_ALERT_FRAMES; окклюзия VISIBILITY_THRESHOLD и fallback FALLBACK_K."""

    def __init__(self, window_size: int, min_alert_frames: int, fallback_k: int):
        self._window_size = window_size
        self._min_alert_frames = min_alert_frames
        self._fallback_k = fallback_k
        self._deque: deque[int] = deque(maxlen=window_size)
        self._occlusion_count = 0
        self._last_valid_state = "NORMAL"

    def update(self, is_abnormal_raw: bool, is_occluded: bool) -> str:
        if is_occluded:
            self._occlusion_count += 1
            if self._occlusion_count > self._fallback_k:
                return "UNRELIABLE"
            return self._last_valid_state

        self._occlusion_count = 0
        flag = int(bool(is_abnormal_raw))
        self._deque.append(flag)
        agg = sum(self._deque)
        if agg >= self._min_alert_frames:
            new_state = "ABNORMAL_FLOOR_POSTURE"
        else:
            new_state = "NORMAL"
        self._last_valid_state = new_state
        return new_state

def run_inference_pipeline() -> pd.DataFrame:
    if not GT_CSV_PATH.exists():
        raise FileNotFoundError(GT_CSV_PATH)

    df_gt_full = pd.read_csv(GT_CSV_PATH).sort_values(
        ["video_name", "frame_index"], kind="mergesort"
    ).reset_index(drop=True)

    df_gt = downsample_gt_per_video(df_gt_full)

    logger.warning(
        "Downsample GT: %d строк -> %d (notfall ~1/4: каждый 4-й; fall ~2/3: 2 из 3 по порядку; прочие — без изменений)",
        len(df_gt_full),
        len(df_gt),
    )

    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    rows_out: list[dict[str, object]] = []
    stabilizer_per_video: dict[str, TemporalStabilizer] = {}

    for row in tqdm(
        df_gt.itertuples(index=False),
        total=len(df_gt),
        desc="YOLO inference (streamed loads)",
        mininterval=1.0,
    ):
        video_name = str(row.video_name)
        frame_index = int(row.frame_index)
        gt_class = str(row.gt_class).strip()

        path = frame_file_path(video_name, frame_index)
        frame_bgr = load_image(path)

        if video_name not in stabilizer_per_video:
            stabilizer_per_video[video_name] = TemporalStabilizer(
                window_size=config.WINDOW_SIZE,
                min_alert_frames=config.MIN_ALERT_FRAMES,
                fallback_k=config.FALLBACK_K,
            )

        inf = infer_single_pose(model, frame_bgr)

        theta = float(inf["theta"])  # type: ignore[arg-type]
        k_ar = float(inf["k_ar"])  # type: ignore[arg-type]
        vis = float(inf["visibility_rate"])  # type: ignore[arg-type]
        is_raw = bool(inf["is_abnormal_raw"])
        is_occluded = vis < float(config.VISIBILITY_THRESHOLD)

        if not inf["ok"] and math.isnan(theta):
            is_raw = False
            is_occluded = True

        stabilizer = stabilizer_per_video[video_name]
        final_status = stabilizer.update(is_raw, is_occluded)
        alert_triggered = final_status == "ABNORMAL_FLOOR_POSTURE"

        rows_out.append(
            {
                "video_name": video_name,
                "frame_index": frame_index,
                "theta": theta,
                "K_ar": float(k_ar),
                "visibility_rate": vis if inf["ok"] else 0.0,
                "gt_class": gt_class,
                "is_abnormal_raw": is_raw,
                "final_status": final_status,
                "alert_triggered": alert_triggered,
            }
        )

    pred_df = pd.DataFrame(rows_out)
    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="utf-8")

    logger.warning(
        "Готово: predictions=%s строк записано в %s",
        len(pred_df),
        PREDICTIONS_CSV_PATH,
    )
    return pred_df

def main() -> None:
    run_inference_pipeline()

if __name__ == "__main__":
    main()
