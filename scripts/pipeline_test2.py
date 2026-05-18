"""test2: YOLO-pose, theta/K, rule classifier, MS temporal filter, occlusion fallback."""
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

TEST2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TEST2_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

FRAMES_DIR = TEST2_ROOT / "data" / "frames"
GT_CSV_PATH = TEST2_ROOT / "data" / "gt" / "frame_level_gt_test2.csv"
PREDICTIONS_CSV_PATH = TEST2_ROOT / "data" / "gt" / "predictions_test2.csv"

WINDOW_MS = 500

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
    visibility_rate = float(np.mean(kpt_conf >= float(config.CONFIDENCE_THRESHOLD)))
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

def temporal_majority_abnormal(buffer: deque[tuple[float, bool]], current_ts: float, window_ms: float) -> bool:
    window_sec = window_ms / 1000.0
    while buffer and current_ts - buffer[0][0] > window_sec:
        buffer.popleft()
    n = len(buffer)
    if n == 0:
        return False
    true_count = sum(1 for _, val in buffer if val)
    if n == 1:
        return bool(buffer[-1][1])
    thresh = int(0.7 * n)
    return true_count >= thresh

class TemporalMsStabilizer:
    """Occlusion fallback; deque (t_sec, is_abn_raw); NORMAL | ABNORMAL_FLOOR_POSTURE | UNRELIABLE."""

    def __init__(self, window_ms: float, fallback_k: int):
        self._window_ms = float(window_ms)
        self._fallback_k = int(fallback_k)
        self._buffer: deque[tuple[float, bool]] = deque()
        self._occlusion_count = 0
        self._last_valid_state = "NORMAL"

    def update(self, timestamp_sec: float, is_abnormal_raw: bool, is_occluded: bool) -> str:
        if is_occluded:
            self._occlusion_count += 1
            if self._occlusion_count > self._fallback_k:
                return "UNRELIABLE"
            return self._last_valid_state

        self._occlusion_count = 0
        self._buffer.append((float(timestamp_sec), bool(is_abnormal_raw)))
        temporal_abn = temporal_majority_abnormal(self._buffer, float(timestamp_sec), self._window_ms)
        new_state = "ABNORMAL_FLOOR_POSTURE" if temporal_abn else "NORMAL"
        self._last_valid_state = new_state
        return new_state

def run_inference_pipeline() -> pd.DataFrame:
    if not GT_CSV_PATH.exists():
        raise FileNotFoundError(GT_CSV_PATH)

    df_gt = pd.read_csv(GT_CSV_PATH).sort_values(
        ["video_name", "frame_index"], kind="mergesort"
    ).reset_index(drop=True)

    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    rows_out: list[dict[str, object]] = []
    stabilizer_per_video: dict[str, TemporalMsStabilizer] = {}

    for row in tqdm(
        df_gt.itertuples(index=False),
        total=len(df_gt),
        desc="YOLO inference test2",
        mininterval=1.0,
    ):
        video_name = str(row.video_name)
        frame_index = int(row.frame_index)
        gt_class = str(row.gt_class).strip()
        try:
            timestamp_sec = float(row.timestamp_sec)
        except AttributeError:
            timestamp_sec = float(frame_index)

        path = frame_file_path(video_name, frame_index)
        frame_bgr = load_image(path)

        if video_name not in stabilizer_per_video:
            stabilizer_per_video[video_name] = TemporalMsStabilizer(
                window_ms=WINDOW_MS,
                fallback_k=int(config.FALLBACK_K),
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
        final_status = stabilizer.update(timestamp_sec, is_raw, is_occluded)
        alert_triggered = final_status == "ABNORMAL_FLOOR_POSTURE"

        rows_out.append(
            {
                "video_name": video_name,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
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

    logger.warning("test2: predictions записаны: %s (%d строк)", PREDICTIONS_CSV_PATH, len(pred_df))
    return pred_df

def main() -> None:
    run_inference_pipeline()

if __name__ == "__main__":
    main()
