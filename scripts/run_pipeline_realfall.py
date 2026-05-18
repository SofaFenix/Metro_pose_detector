"""realfall.mp4 pipeline: YOLO, rule classifier, TemporalStabilizerMs."""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
_FINAL_TEST_ROOT = PROJECT_ROOT.parent
_REPO_ROOT = _FINAL_TEST_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

from temporal_stabilizer_ms import STATUS_ABNORMAL as STAB_HORIZONTAL  # noqa: E402
from temporal_stabilizer_ms import TemporalStabilizerMs  # noqa: E402

LABEL_ABNORMAL_DEMO = "ABNORMAL_FLOOR_POSTURE"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FRAMES_DIR = _FINAL_TEST_ROOT / "data" / "frames" / "realfall"
META_PATH = _FINAL_TEST_ROOT / "data" / "gt" / "realfall_meta.json"
OUT_PRED = _FINAL_TEST_ROOT / "data" / "predictions_realfall.csv"

def load_image(image_path: Path):
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

def infer_single_pose(model: YOLO, frame_bgr, conf_thresh: float | None = None) -> dict[str, object]:
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
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
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

def raw_abnormal(theta: float, k_ar: float) -> bool:
    if math.isnan(theta) or math.isnan(k_ar):
        return False
    return (theta > float(config.THETA_THRESHOLD) and k_ar > float(config.K_BASE)) or (
        k_ar > float(config.K_CRITICAL)
    )

def map_final_label(stabilizer_label: str) -> str:
    if stabilizer_label == STAB_HORIZONTAL:
        return LABEL_ABNORMAL_DEMO
    return stabilizer_label

def resolve_frame_indices() -> tuple[list[int], float]:
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        total = int(meta.get("total_frames") or 0)
        fps = float(meta.get("fps") or 30.0)
        if total > 0:
            return list(range(total)), fps

    paths = sorted(FRAMES_DIR.glob("realfall_frame_*.jpg"))
    indices: list[int] = []
    for p in paths:
        stem = p.stem
        try:
            idx_str = stem.replace("realfall_frame_", "")
            indices.append(int(idx_str))
        except ValueError:
            logger.warning("Не распознан индекс кадра: %s", p.name)
    indices.sort()
    return indices, 30.0

def main() -> None:
    t0 = time.perf_counter()
    indices, fps_use = resolve_frame_indices()
    if not indices:
        logger.warning("Нет кадров в %s — сначала extract_realfall_frames.py", FRAMES_DIR)
        return

    model = YOLO(config.DEFAULT_MODEL, verbose=False)
    stab = TemporalStabilizerMs(
        float(config.TEMPORAL_WINDOW_MS),
        float(config.MIN_ALERT_RATIO),
        int(config.FALLBACK_K),
    )

    rows: list[dict[str, object]] = []
    for frame_index in indices:
        timestamp_sec = round(frame_index / float(fps_use), 6) if fps_use > 0 else float(frame_index)
        path = FRAMES_DIR / f"realfall_frame_{frame_index:05d}.jpg"
        bgr = load_image(path)

        inf = infer_single_pose(model, bgr)
        theta = float(inf["theta"])  # type: ignore[arg-type]
        k_ar = float(inf["k_ar"])  # type: ignore[arg-type]
        vis = float(inf["visibility_rate"])  # type: ignore[arg-type]
        is_raw = raw_abnormal(theta, k_ar)
        if not bool(inf["ok"]) or math.isnan(theta) or math.isnan(k_ar):
            vis_eff = min(vis, float(config.VISIBILITY_THRESHOLD) - 1e-6)
            is_raw = False
        else:
            vis_eff = vis

        raw_stab = stab.update(timestamp_sec, is_raw, vis_eff)
        final_status = map_final_label(raw_stab)
        alert_triggered = 1 if final_status == LABEL_ABNORMAL_DEMO else 0

        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "theta": theta,
                "K_ar": k_ar,
                "visibility_rate": vis if inf["ok"] else 0.0,
                "is_abn_raw": bool(is_raw),
                "final_status": final_status,
                "alert_triggered": alert_triggered,
            }
        )

    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PRED, index=False, encoding="utf-8")
    logger.warning(
        "predictions_realfall.csv: %d строк, время %.2fs",
        len(rows),
        time.perf_counter() - t0,
    )

if __name__ == "__main__":
    main()
