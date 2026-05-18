"""final_test: features CSV or YOLO, rule classifier, TemporalStabilizerMs."""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_TEST_ROOT = PROJECT_ROOT.parent
_REPO_ROOT = FINAL_TEST_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

from temporal_stabilizer_ms import STATUS_ABNORMAL, TemporalStabilizerMs  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = FINAL_TEST_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
GT_CSV_PATH = DATA_DIR / "gt.csv"
FEATURES_CSV_PATH = DATA_DIR / "features.csv"
METADATA_JSON_PATH = DATA_DIR / "metadata.json"
OUT_PREDICTIONS_PATH = DATA_DIR / "predictions_final.csv"

def load_fps_table() -> dict[str, float]:
    if not METADATA_JSON_PATH.exists():
        logger.warning("Нет metadata.json — FPS по умолчанию 30.0")
        return {}
    import json

    raw = json.loads(METADATA_JSON_PATH.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for key, entry in raw.items():
        fps = float(entry.get("fps") or 0.0)
        out[str(key)] = fps if fps > 0 else 30.0
        stem = str(entry.get("stem") or Path(key).stem).lower()
        out.setdefault(stem, out[str(key)])
    return out

def fps_for_video(video_name: str, fps_table: dict[str, float]) -> float:
    vn = str(video_name)
    if vn in fps_table:
        return float(fps_table[vn])
    stem = Path(vn).stem.lower()
    if stem in fps_table:
        return float(fps_table[stem])
    for k, v in fps_table.items():
        if Path(k).stem.lower() == stem:
            return float(v)
    return 30.0

def frame_file_path(video_name: str, frame_index: int) -> Path:
    stem = Path(video_name).stem
    return FRAMES_DIR / f"{stem}_frame_{frame_index:05d}.jpg"

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

def build_feature_frame(df_gt: pd.DataFrame) -> pd.DataFrame:
    """Строки признаков: объединение gt с features.csv или полный инференс."""
    model = None

    if FEATURES_CSV_PATH.exists():
        df_f = pd.read_csv(FEATURES_CSV_PATH)
        cols = {c.lower(): c for c in df_f.columns}
        req_map = {"theta": None, "k_ar": None, "visibility_rate": None}
        for key in req_map:
            if key in cols:
                req_map[key] = cols[key]
            elif key == "k_ar" and "kar" in cols:
                req_map[key] = cols["kar"]
        rename_k = req_map["k_ar"] if req_map["k_ar"] else "K_ar"
        theta_c = req_map["theta"] or "theta"
        vis_c = req_map["visibility_rate"] or "visibility_rate"

        keep = ["video_name", "frame_index", theta_c, rename_k, vis_c]
        keep = [c for c in keep if c in df_f.columns]
        df_f = df_f[keep].copy()
        if rename_k != "K_ar":
            df_f = df_f.rename(columns={rename_k: "K_ar"})
        if theta_c != "theta":
            df_f = df_f.rename(columns={theta_c: "theta"})
        if vis_c != "visibility_rate":
            df_f = df_f.rename(columns={vis_c: "visibility_rate"})

        df = df_gt.merge(df_f, on=["video_name", "frame_index"], how="left")
        needs_infer = df["theta"].isna() | df["K_ar"].isna() | df["visibility_rate"].isna()
        infer_indices = np.flatnonzero(needs_infer.to_numpy())
        logger.warning(
            "features.csv: заполнено %d строк; инференс для %d пропусков",
            int(len(df) - len(infer_indices)),
            int(len(infer_indices)),
        )
    else:
        df = df_gt.copy()
        df["theta"] = np.nan
        df["K_ar"] = np.nan
        df["visibility_rate"] = np.nan
        infer_indices = np.arange(len(df))

    if len(infer_indices) > 0:
        model = YOLO(config.DEFAULT_MODEL, verbose=False)

    rows_theta = df["theta"].to_numpy(copy=True)
    rows_k = df["K_ar"].to_numpy(copy=True)
    rows_vis = df["visibility_rate"].to_numpy(copy=True)

    for idx in tqdm(infer_indices, desc="YOLO inference (final_test gaps)", mininterval=1.0):
        row = df.iloc[int(idx)]
        path = frame_file_path(str(row["video_name"]), int(row["frame_index"]))
        bgr = load_image(path)
        inf = infer_single_pose(model, bgr)
        rows_theta[int(idx)] = float(inf["theta"]) if inf["ok"] else float("nan")
        rows_k[int(idx)] = float(inf["k_ar"])
        rows_vis[int(idx)] = float(inf["visibility_rate"]) if inf["ok"] else 0.0

    df = df.copy()
    df["theta"] = rows_theta
    df["K_ar"] = rows_k
    df["visibility_rate"] = rows_vis
    return df

def run_pipeline() -> pd.DataFrame:
    if not GT_CSV_PATH.exists():
        raise FileNotFoundError(GT_CSV_PATH)

    df_gt = pd.read_csv(GT_CSV_PATH).sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(
        drop=True
    )
    fps_table = load_fps_table()

    df_feat = build_feature_frame(df_gt)

    stabilizers: dict[str, TemporalStabilizerMs] = {}
    window_ms = float(config.TEMPORAL_WINDOW_MS)
    min_ratio = float(config.MIN_ALERT_RATIO)
    fb_k = int(config.FALLBACK_K)

    rows_out: list[dict[str, object]] = []

    for row in tqdm(df_feat.itertuples(index=False), total=len(df_feat), desc="Temporal filter", mininterval=1.0):
        video_name = str(getattr(row, "video_name"))
        frame_index = int(getattr(row, "frame_index"))
        gt_class = str(getattr(row, "gt_class")).strip()

        fps = fps_for_video(video_name, fps_table)
        timestamp_sec = round(frame_index / float(fps), 6) if fps > 0 else float(frame_index)

        theta = float(getattr(row, "theta"))
        k_ar = float(getattr(row, "K_ar"))
        vis = float(getattr(row, "visibility_rate"))

        if video_name not in stabilizers:
            stabilizers[video_name] = TemporalStabilizerMs(window_ms, min_ratio, fb_k)

        is_abn_raw = raw_abnormal(theta, k_ar)
        if math.isnan(theta) or math.isnan(k_ar):
            vis_eff = min(vis, float(config.VISIBILITY_THRESHOLD) - 1e-6)
        else:
            vis_eff = vis

        final_status = stabilizers[video_name].update(timestamp_sec, is_abn_raw, vis_eff)
        alert_triggered = 1 if final_status == STATUS_ABNORMAL else 0

        rows_out.append(
            {
                "video_name": video_name,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "theta": theta,
                "K_ar": k_ar,
                "visibility_rate": vis,
                "gt_class": gt_class,
                "is_abn_raw": bool(is_abn_raw),
                "final_status": final_status,
                "alert_triggered": alert_triggered,
            }
        )

    pred_df = pd.DataFrame(rows_out)
    OUT_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT_PREDICTIONS_PATH, index=False, encoding="utf-8")
    logger.warning("predictions_final.csv записан: %s (%d строк)", OUT_PREDICTIONS_PATH, len(pred_df))
    return pred_df

def main() -> None:
    run_pipeline()

if __name__ == "__main__":
    main()
