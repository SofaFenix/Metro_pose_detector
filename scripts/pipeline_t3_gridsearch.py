"""Pipeline pass for temporal grid search dataset."""
from __future__ import annotations

import logging
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

FRAMES_ROOT = T3_ROOT / "frames"
GT_CSV_PATH = T3_ROOT / "gt" / "frame_level_gt.csv"
RAW_FEATURES_PATH = T3_ROOT / "results" / "raw_features_t3.csv"

STATUS_NORMAL = "NORMAL"
STATUS_ABNORMAL = "ABNORMAL_HORIZONTAL_POSTURE"
STATUS_UNRELIABLE = "UNRELIABLE"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def frame_file_path(video_name: str, frame_index: int) -> Path:
    stem = Path(video_name).stem
    return FRAMES_ROOT / stem / f"frame_{frame_index:05d}.jpg"

def load_image(image_path: Path) -> np.ndarray | None:
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

def compute_theta_deg(keypoints_xy: np.ndarray) -> float:
    shoulders = (keypoints_xy[5] + keypoints_xy[6]) / 2.0
    hips = (keypoints_xy[11] + keypoints_xy[12]) / 2.0
    return abs(math.degrees(math.atan2((hips[0] - shoulders[0]), (hips[1] - shoulders[1]))))

def infer_single_pose(
    model: YOLO,
    frame_bgr: np.ndarray | None,
    conf_thresh: float | None = None,
) -> dict[str, object]:
    if conf_thresh is None:
        conf_thresh = float(config.CONFIDENCE_THRESHOLD)

    empty: dict[str, object] = {
        "ok": False,
        "theta": float("nan"),
        "k_ar": float("nan"),
        "visibility_rate": 0.0,
        "is_abnormal_raw": False,
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
        return {**empty, "k_ar": k_ar}

    kpt_xy = res.keypoints.xy[best_idx].detach().cpu().numpy()
    kpt_conf = res.keypoints.conf[best_idx].detach().cpu().numpy()
    theta = compute_theta_deg(kpt_xy)
    visibility_rate = float(np.mean(kpt_conf >= float(config.CONFIDENCE_THRESHOLD)))
    raw = (theta > float(config.THETA_THRESHOLD) and k_ar > float(config.K_BASE)) or (
        k_ar > float(config.K_CRITICAL)
    )

    return {
        "ok": True,
        "theta": float(theta),
        "k_ar": float(k_ar),
        "visibility_rate": visibility_rate,
        "is_abnormal_raw": bool(raw),
    }

def temporal_majority_abnormal(
    buffer: deque[tuple[float, bool]],
    current_ts: float,
    window_ms: float,
    min_alert_ratio: float,
) -> bool:
    window_sec = window_ms / 1000.0
    while buffer and current_ts - buffer[0][0] > window_sec:
        buffer.popleft()
    n = len(buffer)
    if n == 0:
        return False
    true_count = sum(1 for _, val in buffer if val)
    if n == 1:
        return bool(buffer[-1][1])
    need = max(1, int(math.ceil(min_alert_ratio * n - 1e-9)))
    return true_count >= need

class TemporalStabilizer:
    """Окклюзия + fallback (FALLBACK_K) + временное окно в миллисекундах."""

    def __init__(self, window_ms: float, min_alert_ratio: float, fallback_k: int):
        self._window_ms = float(window_ms)
        self._min_alert_ratio = float(min_alert_ratio)
        self._fallback_k = int(fallback_k)
        self._buffer: deque[tuple[float, bool]] = deque()
        self._occlusion_count = 0
        self._last_valid_state = STATUS_NORMAL

    def update(self, timestamp_sec: float, is_abnormal_raw: bool, is_occluded: bool) -> str:
        if is_occluded:
            self._occlusion_count += 1
            if self._occlusion_count > self._fallback_k:
                return STATUS_UNRELIABLE
            return self._last_valid_state

        self._occlusion_count = 0
        self._buffer.append((float(timestamp_sec), bool(is_abnormal_raw)))
        temporal_abn = temporal_majority_abnormal(
            self._buffer,
            float(timestamp_sec),
            self._window_ms,
            self._min_alert_ratio,
        )
        new_state = STATUS_ABNORMAL if temporal_abn else STATUS_NORMAL
        self._last_valid_state = new_state
        return new_state

def run_raw_feature_extraction(
    gt_df: pd.DataFrame | None = None,
    *,
    save_path: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """YOLO + правило по config; без временного фильтра (для перебора гиперпараметров)."""
    out_path = save_path or RAW_FEATURES_PATH
    if use_cache and out_path.is_file():
        return pd.read_csv(out_path)

    if gt_df is None:
        if not GT_CSV_PATH.exists():
            raise FileNotFoundError(GT_CSV_PATH)
        gt_df = pd.read_csv(GT_CSV_PATH)

    gt_df = gt_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
    model = YOLO(config.DEFAULT_MODEL, verbose=False)
    vis_thresh = float(config.VISIBILITY_THRESHOLD)

    rows: list[dict[str, object]] = []
    for row in tqdm(gt_df.itertuples(index=False), total=len(gt_df), desc="YOLO raw features t3"):
        video_name = str(row.video_name)
        frame_index = int(row.frame_index)
        timestamp_sec = float(row.timestamp_sec)
        gt_class = str(row.gt_class).strip()

        path = frame_file_path(video_name, frame_index)
        frame_bgr = load_image(path)
        inf = infer_single_pose(model, frame_bgr)

        vis = float(inf["visibility_rate"]) if inf["ok"] else 0.0
        is_raw = bool(inf["is_abnormal_raw"]) if inf["ok"] else False
        is_occluded = vis < vis_thresh
        if not inf["ok"]:
            is_raw = False
            is_occluded = True

        rows.append(
            {
                "video_name": video_name,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "timestamp_ms": round(timestamp_sec * 1000.0, 3),
                "gt_class": gt_class,
                "visibility_rate": vis,
                "is_abnormal_raw": is_raw,
                "is_occluded": is_occluded,
            }
        )

    raw_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_path, index=False, encoding="utf-8")
    return raw_df

def apply_temporal_stabilizer(
    raw_df: pd.DataFrame,
    *,
    window_ms: float,
    min_alert_ratio: float,
    fallback_k: int | None = None,
) -> pd.DataFrame:
    """Применяет TemporalStabilizer к кэшу сырых признаков."""
    if fallback_k is None:
        fallback_k = int(config.FALLBACK_K)

    raw_df = raw_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    stabilizers: dict[str, TemporalStabilizer] = {}

    for row in raw_df.itertuples(index=False):
        video_name = str(row.video_name)
        if video_name not in stabilizers:
            stabilizers[video_name] = TemporalStabilizer(
                window_ms=window_ms,
                min_alert_ratio=min_alert_ratio,
                fallback_k=fallback_k,
            )
        stab = stabilizers[video_name]
        ts = float(row.timestamp_sec)
        final_status = stab.update(ts, bool(row.is_abnormal_raw), bool(row.is_occluded))
        alert = final_status == STATUS_ABNORMAL

        rows.append(
            {
                "video_name": video_name,
                "frame_index": int(row.frame_index),
                "timestamp_ms": float(row.timestamp_ms),
                "final_status": final_status,
                "alert_triggered": alert,
                "gt_class": str(row.gt_class),
            }
        )

    return pd.DataFrame(rows)

def run_pipeline_with_params(
    window_ms: float,
    min_alert_ratio: float,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw_df is None:
        raw_df = run_raw_feature_extraction()
    return apply_temporal_stabilizer(
        raw_df,
        window_ms=window_ms,
        min_alert_ratio=min_alert_ratio,
        fallback_k=int(config.FALLBACK_K),
    )
