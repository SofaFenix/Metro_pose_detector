"""realfall overlays and demo video from predictions_realfall.csv."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
_FINAL_TEST_ROOT = PROJECT_ROOT.parent
_REPO_ROOT = _FINAL_TEST_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

from run_pipeline_realfall import LABEL_ABNORMAL_DEMO  # noqa: E402
from run_pipeline_realfall import infer_single_pose  # noqa: E402
from run_pipeline_realfall import load_image  # noqa: E402

FRAMES_DIR = _FINAL_TEST_ROOT / "data" / "frames" / "realfall"
META_PATH = _FINAL_TEST_ROOT / "data" / "gt" / "realfall_meta.json"
PRED_PATH = _FINAL_TEST_ROOT / "data" / "predictions_realfall.csv"
FIG_DIR = _FINAL_TEST_ROOT / "figures" / "realfall_demo"
VIDEO_OUT = _FINAL_TEST_ROOT / "figures" / "realfall_full_demo.mp4"

INCIDENT_SEC = 3.10
VIS_FOCUS_SEC = 2.50

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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def read_fps() -> float:
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        fps = float(meta.get("fps") or 0.0)
        if fps > 1e-6:
            return fps
    return 30.0

def status_color(status: str) -> tuple[int, int, int]:
    if status == "NORMAL":
        return (0, 200, 0)
    if status == LABEL_ABNORMAL_DEMO:
        return (0, 0, 255)
    return (128, 128, 128)

def blend_alert_red(frame_drawn: np.ndarray) -> np.ndarray:
    overlay = np.zeros_like(frame_drawn)
    overlay[:, :] = (0, 0, 200)
    return cv2.addWeighted(frame_drawn, 0.65, overlay, 0.35, 0)

def draw_pose_overlay(
    canvas: np.ndarray,
    inf: dict[str, object],
    *,
    final_status: str,
    alert_triggered: bool,
    timestamp_sec: float,
    mean_conf_label: float | None,
) -> np.ndarray:
    out = canvas.copy()
    h, w_img = out.shape[:2]
    color = status_color(final_status)

    bbox = inf.get("bbox_xyxy")
    if isinstance(bbox, tuple) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox  # type: ignore[misc]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    kp = inf.get("keypoints_xy")
    if isinstance(kp, np.ndarray) and kp.shape[0] >= 17:
        for a, b in COCO_EDGES:
            if int(a) >= len(kp) or int(b) >= len(kp):
                continue
            pa = tuple(map(int, kp[a]))
            pb = tuple(map(int, kp[b]))
            cv2.line(out, pa, pb, (200, 200, 200), 2)
        for idx in range(17):
            x, y = map(int, kp[idx])
            cv2.circle(out, (x, y), 3, color, -1)

    mc = mean_conf_label if mean_conf_label is not None else float(inf.get("mean_conf", 0.0) or 0.0)
    line1 = f"status={final_status}"
    line2 = f"mean_conf={mc:.3f}"
    line3 = f"t={timestamp_sec:.2f}s alert={int(alert_triggered)}"

    y0 = 28
    panel_w, panel_h = min(520, w_img - 8), min(110, h - 8)
    cv2.rectangle(out, (6, 6), (6 + panel_w, 6 + panel_h), (12, 12, 12), -1)
    cv2.rectangle(out, (6, 6), (6 + panel_w, 6 + panel_h), color, 2)
    cv2.putText(out, line1, (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(out, line2, (14, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(out, line3, (14, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 2, cv2.LINE_AA)

    if alert_triggered:
        cv2.putText(
            out,
            "ALERT: ABNORMAL_FLOOR_POSTURE",
            (14, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"t={timestamp_sec:.2f}s",
        (w_img - 170, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"{final_status}",
        (w_img - 260, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return out

def pick_snapshot_indices(df: pd.DataFrame) -> list[tuple[int, str]]:
    df = df.sort_values("frame_index").reset_index(drop=True)
    out: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(ix: int, tag: str) -> None:
        if ix not in seen:
            out.append((ix, tag))
            seen.add(ix)

    before = df[(df["timestamp_sec"] < INCIDENT_SEC - 0.08) & (df["final_status"] == "NORMAL")]
    if len(before) >= 2:
        picks = np.linspace(0, len(before) - 1, num=2, dtype=int)
        for pi in picks:
            add(int(before.iloc[int(pi)]["frame_index"]), "before_normal")
    elif len(before) == 1:
        add(int(before.iloc[0]["frame_index"]), "before_normal")

    during = df[
        (df["timestamp_sec"] >= INCIDENT_SEC)
        & ((df["final_status"] == LABEL_ABNORMAL_DEMO) | (df["alert_triggered"].astype(int) == 1))
    ]
    if len(during) >= 2:
        picks = np.linspace(0, len(during) - 1, num=2, dtype=int)
        for pi in picks:
            add(int(during.iloc[int(pi)]["frame_index"]), "during_abnormal")
    elif len(during) == 1:
        add(int(during.iloc[0]["frame_index"]), "during_abnormal")

    unrel = df[df["final_status"] == "UNRELIABLE"]
    if len(unrel) > 0:
        mid = int(unrel.iloc[len(unrel) // 2]["frame_index"])
        add(mid, "unreliable")
    else:
        low_vis = df[df["visibility_rate"] < float(config.VISIBILITY_THRESHOLD)]
        if len(low_vis) > 0:
            mid = int(low_vis.iloc[len(low_vis) // 2]["frame_index"])
            add(mid, "low_visibility_standin")

    if len(df) > 0:
        bi = int((df["timestamp_sec"] - INCIDENT_SEC).abs().idxmin())
        add(int(df.loc[bi, "frame_index"]), "boundary_incident")

    before_out = len(out)
    if len(out) < 5 and len(df) > 0:
        mid_i = int(df.iloc[len(df) // 2]["frame_index"])
        add(mid_i, "extra_fill")
        if len(out) == before_out:
            add(int(df.iloc[0]["frame_index"]), "extra_head")

    return out[:7]

def main() -> None:
    t0 = time.perf_counter()
    fps_use = read_fps()

    if not PRED_PATH.is_file():
        logger.warning("Нет %s — сначала run_pipeline_realfall.py", PRED_PATH)
        return

    df = pd.read_csv(PRED_PATH)
    if df.empty:
        logger.warning("Пустой predictions_realfall.csv")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pred_by_fi = df.set_index("frame_index")
    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    picks = pick_snapshot_indices(df)
    for seq, (fi, tag) in enumerate(picks):
        path = FRAMES_DIR / f"realfall_frame_{fi:05d}.jpg"
        row = pred_by_fi.loc[fi]
        ts = float(row["timestamp_sec"])
        status = str(row["final_status"])
        alert = int(row["alert_triggered"]) == 1

        bgr = load_image(path)
        if bgr is None:
            logger.warning("Пропуск snapshot frame_index=%d (нет файла)", fi)
            continue
        try:
            inf = infer_single_pose(model, bgr)
        except Exception as exc:
            logger.warning("infer snapshot fi=%d: %s", fi, exc)
            inf = {"ok": False, "bbox_xyxy": None, "keypoints_xy": None, "mean_conf": 0.0}

        mc = float(inf["mean_conf"]) if inf.get("ok") else float(row.get("visibility_rate", 0.0))
        drawn = draw_pose_overlay(
            bgr,
            inf,
            final_status=status,
            alert_triggered=alert,
            timestamp_sec=ts,
            mean_conf_label=mc,
        )
        if alert:
            drawn = blend_alert_red(drawn)

        out_jpg = FIG_DIR / f"realfall_snap_{seq + 1:02d}_{tag}_fi{fi}.jpg"
        ok_enc, buf = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        if ok_enc:
            buf.tofile(str(out_jpg))

    segment_df = df[df["timestamp_sec"] >= VIS_FOCUS_SEC].sort_values("frame_index")
    if segment_df.empty:
        logger.warning("Нет кадров для сегмента >= %.2f с", VIS_FOCUS_SEC)
        return

    first_row = segment_df.iloc[0]
    probe_path = FRAMES_DIR / f"realfall_frame_{int(first_row['frame_index']):05d}.jpg"
    probe = load_image(probe_path)
    if probe is None:
        logger.warning("Не удалось прочитать пробный кадр для видео")
        return
    h, w = probe.shape[:2]

    VIDEO_OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(VIDEO_OUT), cv2.VideoWriter_fourcc(*"mp4v"), float(fps_use), (w, h))

    for _, row in segment_df.iterrows():
        fi = int(row["frame_index"])
        ts = float(row["timestamp_sec"])
        status = str(row["final_status"])
        alert = int(row["alert_triggered"]) == 1

        path = FRAMES_DIR / f"realfall_frame_{fi:05d}.jpg"
        bgr = load_image(path)
        if bgr is None:
            logger.warning("Пропуск видео frame_index=%d", fi)
            continue

        try:
            inf = infer_single_pose(model, bgr)
        except Exception as exc:
            logger.warning("infer video fi=%d: %s", fi, exc)
            inf = {"ok": False, "bbox_xyxy": None, "keypoints_xy": None, "mean_conf": 0.0}

        mc = float(inf["mean_conf"]) if inf.get("ok") else 0.0
        frame_use = cv2.resize(bgr, (w, h)) if (bgr.shape[1], bgr.shape[0]) != (w, h) else bgr
        drawn = draw_pose_overlay(
            frame_use,
            inf,
            final_status=status,
            alert_triggered=alert,
            timestamp_sec=ts,
            mean_conf_label=mc,
        )
        if alert:
            drawn = blend_alert_red(drawn)
            cv2.putText(
                drawn,
                "ALERT: ABNORMAL_FLOOR_POSTURE",
                (14, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.putText(
            drawn,
            f"focus >= {VIS_FOCUS_SEC:.2f}s",
            (14, h - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        writer.write(drawn)

    writer.release()
    logger.warning(
        "Визуализация realfall завершена за %.2fs — видео %s",
        time.perf_counter() - t0,
        VIDEO_OUT.name,
    )

if __name__ == "__main__":
    main()
