"""Shared comparison utilities for test2_with_filters."""
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
T2F_ROOT = PROJECT_ROOT / "test2_with_filters"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

FRAMES_DIR = T2F_ROOT / "frames"
GT_CSV_PATH = T2F_ROOT / "gt" / "frame_level_gt.csv"
RAW_FEATURES_PATH = T2F_ROOT / "results" / "raw_features.csv"

STATUS_NORMAL = "NORMAL"
STATUS_ABNORMAL = "ABNORMAL_FLOOR_POSTURE"
STATUS_UNRELIABLE = "UNRELIABLE"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def frame_file_path(video_name: str, frame_index: int) -> Path:
    stem = Path(video_name).stem
    return FRAMES_DIR / f"{stem}_frame_{frame_index:05d}.jpg"

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

def infer_single_pose(model: YOLO, frame_bgr: np.ndarray | None) -> dict[str, object]:
    conf_thresh = float(config.CONFIDENCE_THRESHOLD)
    empty: dict[str, object] = {
        "ok": False,
        "visibility_rate": 0.0,
        "is_abnormal_raw": False,
    }
    if frame_bgr is None or frame_bgr.size == 0:
        return empty
    try:
        results = model.predict(frame_bgr, conf=conf_thresh, verbose=False)
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
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    k_ar = float(w / h)
    if res.keypoints is None or len(res.keypoints) <= best_idx:
        return empty
    kpt_xy = res.keypoints.xy[best_idx].detach().cpu().numpy()
    kpt_conf = res.keypoints.conf[best_idx].detach().cpu().numpy()
    theta = compute_theta_deg(kpt_xy)
    visibility_rate = float(np.mean(kpt_conf >= float(config.CONFIDENCE_THRESHOLD)))
    raw = (theta > float(config.THETA_THRESHOLD) and k_ar > float(config.K_BASE)) or (
        k_ar > float(config.K_CRITICAL)
    )
    return {
        "ok": True,
        "visibility_rate": visibility_rate,
        "is_abnormal_raw": bool(raw),
        "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
        "keypoints_xy": kpt_xy.astype(np.float32),
        "keypoints_conf": kpt_conf.astype(np.float32),
        "mean_conf": float(np.mean(kpt_conf)),
    }

def run_raw_feature_extraction(gt_df: pd.DataFrame | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and RAW_FEATURES_PATH.is_file():
        return pd.read_csv(RAW_FEATURES_PATH)
    if gt_df is None:
        gt_df = pd.read_csv(GT_CSV_PATH)
    gt_df = gt_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
    model = YOLO(config.DEFAULT_MODEL, verbose=False)
    vis_thresh = float(config.VISIBILITY_THRESHOLD)
    rows: list[dict[str, object]] = []
    for row in tqdm(gt_df.itertuples(index=False), total=len(gt_df), desc="YOLO raw (test2 filters)"):
        video_name = str(row.video_name)
        frame_index = int(row.frame_index)
        timestamp_sec = float(row.timestamp_sec)
        path = frame_file_path(video_name, frame_index)
        inf = infer_single_pose(model, load_image(path))
        vis = float(inf["visibility_rate"]) if inf["ok"] else 0.0
        is_raw = bool(inf["is_abnormal_raw"]) if inf["ok"] else False
        is_occluded = vis < vis_thresh or not inf["ok"]
        if not inf["ok"]:
            is_raw = False
        rows.append(
            {
                "video_name": video_name,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "gt_class": str(row.gt_class).strip(),
                "visibility_rate": vis,
                "is_abnormal_raw": is_raw,
                "is_occluded": is_occluded,
            }
        )
    raw_df = pd.DataFrame(rows)
    RAW_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_FEATURES_PATH, index=False, encoding="utf-8")
    return raw_df

class FrameLevelStabilizer:
    """Config A: без временного окна — покадровое решение + fallback."""

    def __init__(self, fallback_k: int):
        self._fallback_k = int(fallback_k)
        self._occlusion_count = 0
        self._last_valid_state = STATUS_NORMAL

    def update(self, is_abnormal_raw: bool, is_occluded: bool) -> str:
        if is_occluded:
            self._occlusion_count += 1
            if self._occlusion_count > self._fallback_k:
                return STATUS_UNRELIABLE
            return self._last_valid_state
        self._occlusion_count = 0
        new_state = STATUS_ABNORMAL if is_abnormal_raw else STATUS_NORMAL
        self._last_valid_state = new_state
        return new_state

class TemporalFrameStabilizer:
    """Config B: фиксированное окно в кадрах (config.WINDOW_SIZE, config.MIN_ALERT_FRAMES)."""

    def __init__(self, window_size: int, min_alert_frames: int, fallback_k: int):
        self._deque: deque[int] = deque(maxlen=int(window_size))
        self._min_alert_frames = int(min_alert_frames)
        self._fallback_k = int(fallback_k)
        self._occlusion_count = 0
        self._last_valid_state = STATUS_NORMAL

    def update(self, is_abnormal_raw: bool, is_occluded: bool) -> str:
        if is_occluded:
            self._occlusion_count += 1
            if self._occlusion_count > self._fallback_k:
                return STATUS_UNRELIABLE
            return self._last_valid_state
        self._occlusion_count = 0
        self._deque.append(int(bool(is_abnormal_raw)))
        if sum(self._deque) >= self._min_alert_frames:
            new_state = STATUS_ABNORMAL
        else:
            new_state = STATUS_NORMAL
        self._last_valid_state = new_state
        return new_state

def temporal_majority_ms(
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
    true_count = sum(1 for _, v in buffer if v)
    if n == 1:
        return bool(buffer[-1][1])
    need = max(1, int(math.ceil(min_alert_ratio * n - 1e-9)))
    return true_count >= need

class TemporalMsStabilizer:
    """Config C: окно в миллисекундах + min_alert_ratio (Grid Search)."""

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
        temporal_abn = temporal_majority_ms(
            self._buffer, float(timestamp_sec), self._window_ms, self._min_alert_ratio
        )
        new_state = STATUS_ABNORMAL if temporal_abn else STATUS_NORMAL
        self._last_valid_state = new_state
        return new_state

def apply_config(raw_df: pd.DataFrame, config_key: str, *, window_ms: float, min_alert_ratio: float) -> pd.DataFrame:
    raw_df = raw_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
    fallback_k = int(config.FALLBACK_K)
    rows: list[dict[str, object]] = []
    stabilizers: dict[str, object] = {}

    for row in raw_df.itertuples(index=False):
        video_name = str(row.video_name)
        if video_name not in stabilizers:
            if config_key == "A":
                stabilizers[video_name] = FrameLevelStabilizer(fallback_k=fallback_k)
            elif config_key == "B":
                stabilizers[video_name] = TemporalFrameStabilizer(
                    window_size=int(config.WINDOW_SIZE),
                    min_alert_frames=int(config.MIN_ALERT_FRAMES),
                    fallback_k=fallback_k,
                )
            else:
                stabilizers[video_name] = TemporalMsStabilizer(
                    window_ms=window_ms,
                    min_alert_ratio=min_alert_ratio,
                    fallback_k=fallback_k,
                )
        stab = stabilizers[video_name]
        is_raw = bool(row.is_abnormal_raw)
        is_occ = bool(row.is_occluded)
        if config_key == "C":
            final_status = stab.update(float(row.timestamp_sec), is_raw, is_occ)  # type: ignore[union-attr]
        else:
            final_status = stab.update(is_raw, is_occ)  # type: ignore[union-attr]
        rows.append(
            {
                "video_name": video_name,
                "frame_index": int(row.frame_index),
                "timestamp_sec": float(row.timestamp_sec),
                "visibility_rate": float(row.visibility_rate),
                "gt_class": str(row.gt_class),
                "is_abnormal_raw": is_raw,
                "final_status": final_status,
                "alert_triggered": final_status == STATUS_ABNORMAL,
            }
        )
    return pd.DataFrame(rows)

# Event-level metrics

def extract_contiguous_intervals(df: pd.DataFrame, active_col: str) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    for video_name, grp in df.groupby("video_name", sort=False):
        grp = grp.sort_values("frame_index", kind="mergesort").reset_index(drop=True)
        flags = grp[active_col].astype(bool).to_numpy()
        in_run = False
        start_i = 0
        for i, active in enumerate(flags):
            if active and not in_run:
                in_run, start_i = True, i
            elif not active and in_run:
                intervals.append(
                    {
                        "video_name": video_name,
                        "start_sec": float(grp.iloc[start_i]["timestamp_sec"]),
                        "end_sec": float(grp.iloc[i - 1]["timestamp_sec"]),
                    }
                )
                in_run = False
        if in_run:
            intervals.append(
                {
                    "video_name": video_name,
                    "start_sec": float(grp.iloc[start_i]["timestamp_sec"]),
                    "end_sec": float(grp.iloc[-1]["timestamp_sec"]),
                }
            )
    return intervals

def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)

def compute_event_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, float]:
    gt_df = gt_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
    pred_df = pred_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)

    gt_tmp = gt_df.copy()
    gt_tmp["_gt_abn"] = gt_tmp["gt_class"] == STATUS_ABNORMAL
    gt_intervals = extract_contiguous_intervals(gt_tmp, "_gt_abn")

    pred_tmp = pred_df.copy()
    pred_tmp["_alert"] = pred_tmp["alert_triggered"].astype(bool)
    alert_runs = extract_contiguous_intervals(pred_tmp, "_alert")

    pred_alerts = pred_df.loc[pred_df["alert_triggered"].astype(bool)]
    detected = 0
    latencies_ms: list[float] = []
    for inc in gt_intervals:
        vid = inc["video_name"]
        s_sec, e_sec = float(inc["start_sec"]), float(inc["end_sec"])
        hits = pred_alerts.loc[
            (pred_alerts["video_name"] == vid)
            & (pred_alerts["timestamp_sec"] >= s_sec)
            & (pred_alerts["timestamp_sec"] <= e_sec)
        ]
        if len(hits) > 0:
            detected += 1
            latencies_ms.append(max(0.0, (float(hits["timestamp_sec"].min()) - s_sec) * 1000.0))

    n_incidents = len(gt_intervals)
    event_recall = detected / n_incidents if n_incidents else 0.0

    gt_by_video: dict[str, list[tuple[float, float]]] = {}
    for inc in gt_intervals:
        gt_by_video.setdefault(inc["video_name"], []).append(
            (float(inc["start_sec"]), float(inc["end_sec"]))
        )
    tp_alerts = 0
    for ar in alert_runs:
        vid, a0, a1 = ar["video_name"], float(ar["start_sec"]), float(ar["end_sec"])
        if any(_interval_overlap(a0, a1, g0, g1) for g0, g1 in gt_by_video.get(vid, [])):
            tp_alerts += 1
    n_alert_events = len(alert_runs)
    event_precision = tp_alerts / n_alert_events if n_alert_events else 0.0
    f1_event = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if (event_precision + event_recall) > 0
        else 0.0
    )
    latency_ms = float(np.mean(latencies_ms)) if latencies_ms else float("nan")

    is_normal = gt_df.groupby("video_name")["gt_class"].apply(lambda s: (s == STATUS_ABNORMAL).sum() == 0)
    false_alerts = 0
    normal_duration_sec = 0.0
    for vid in is_normal[is_normal].index:
        g = gt_df.loc[gt_df["video_name"] == vid]
        if g.empty:
            continue
        normal_duration_sec += float(g["timestamp_sec"].max()) - float(g["timestamp_sec"].min()) + 1.0 / 30.0
        false_alerts += int(
            pred_df.loc[(pred_df["video_name"] == vid) & pred_df["alert_triggered"].astype(bool)].shape[0]
        )
    normal_min = normal_duration_sec / 60.0 if normal_duration_sec > 0 else 0.0
    far = false_alerts / normal_min if normal_min > 0 else 0.0

    vis_thresh = float(config.VISIBILITY_THRESHOLD)
    excluded_visibility = int((pred_df["visibility_rate"] < vis_thresh).sum())

    return {
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "f1_event": float(f1_event),
        "latency_ms": float(latency_ms),
        "far_alerts_per_min": float(far),
        "excluded_visibility_frames": excluded_visibility,
        "total_frames": int(len(pred_df)),
    }
