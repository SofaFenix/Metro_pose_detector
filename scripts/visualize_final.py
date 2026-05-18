"""Visualization: fall4, normal2, normal3."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_TEST_ROOT = PROJECT_ROOT.parent
_REPO_ROOT = FINAL_TEST_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

from pipeline_final import infer_single_pose  # noqa: E402
from temporal_stabilizer_ms import STATUS_ABNORMAL, TemporalStabilizerMs  # noqa: E402

FIGURES_DIR = FINAL_TEST_ROOT / "figures"
METADATA_JSON_PATH = FINAL_TEST_ROOT / "data" / "metadata.json"

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

def candidate_video_paths(stem: str) -> list[Path]:
    stem = stem.strip().lower()
    base = _REPO_ROOT / "data"
    return [
        base / "archive_videos" / f"{stem}.mp4",
        base / "normal" / f"{stem}.mp4",
        base / "archive" / f"{stem}.mp4",
        _REPO_ROOT / f"{stem}.mp4",
        _REPO_ROOT / f"demo_{stem}.mp4",
    ]

def resolve_video_path(stem: str) -> Path | None:
    for p in candidate_video_paths(stem):
        try:
            if p.is_file():
                return p
        except OSError as exc:
            logger.warning("Ошибка доступа к пути %s: %s", p, exc)
    return None

def status_color(status: str) -> tuple[int, int, int]:
    if status == "NORMAL":
        return (0, 200, 0)
    if status == STATUS_ABNORMAL:
        return (0, 0, 255)
    return (128, 128, 128)

def draw_pose_overlay(
    canvas: np.ndarray,
    inf: dict[str, object],
    *,
    final_status: str,
    alert_triggered: bool,
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

    mean_conf = float(inf.get("mean_conf", 0.0) or 0.0)
    line1 = f"status={final_status}"
    line2 = f"mean_conf={mean_conf:.3f}"
    line3 = f"alert_triggered={alert_triggered}"

    y0 = 28
    panel_w, panel_h = min(420, w_img - 8), min(100, h - 8)
    cv2.rectangle(out, (6, 6), (6 + panel_w, 6 + panel_h), (12, 12, 12), -1)
    cv2.rectangle(out, (6, 6), (6 + panel_w, 6 + panel_h), color, 2)
    cv2.putText(out, line1, (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(out, line2, (14, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(out, line3, (14, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
    return out

def blend_alert_red(frame_drawn: np.ndarray) -> np.ndarray:
    overlay = np.zeros_like(frame_drawn)
    overlay[:, :] = (0, 0, 200)
    return cv2.addWeighted(frame_drawn, 0.65, overlay, 0.35, 0)

def read_metadata_fps(stem: str) -> float | None:
    if not METADATA_JSON_PATH.exists():
        return None
    import json

    meta = json.loads(METADATA_JSON_PATH.read_text(encoding="utf-8"))
    stem_low = stem.lower()
    for key, entry in meta.items():
        if Path(key).stem.lower() == stem_low:
            fps = float(entry.get("fps") or 0.0)
            return fps if fps > 0 else None
        if str(entry.get("stem", "")).lower() == stem_low:
            fps = float(entry.get("fps") or 0.0)
            return fps if fps > 0 else None
    return None

def render_video_demo(model: YOLO, stem: str, label: str) -> None:
    src = resolve_video_path(stem)
    if src is None:
        logger.warning("Файл видео для stem=%s не найден (ожидались пути под data/).", stem)
        return

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        logger.warning("Не удалось открыть видео: %s", src)
        return

    fps_prop = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta_fps = read_metadata_fps(stem)
    fps_use = fps_prop if fps_prop > 1e-3 else (meta_fps or 24.0)

    w_i = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_i = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w_i <= 0 or h_i <= 0:
        logger.warning("Некорректный размер кадра: %s", src)
        cap.release()
        return

    stab = TemporalStabilizerMs(
        float(config.TEMPORAL_WINDOW_MS),
        float(config.MIN_ALERT_RATIO),
        int(config.FALLBACK_K),
    )

    out_path = FIGURES_DIR / f"demo_{stem}_final.mp4"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps_use), (w_i, h_i))

    frame_index = 0
    saved_shots = 0
    shot_positions = {0, max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 2) // 3)}

    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        timestamp_sec = frame_index / float(fps_use) if fps_use > 0 else float(frame_index)

        try:
            inf = infer_single_pose(model, bgr)
        except Exception as exc:
            logger.warning("infer_single_pose frame=%d: %s", frame_index, exc)
            inf = {
                "ok": False,
                "bbox_xyxy": None,
                "keypoints_xy": None,
                "mean_conf": 0.0,
                "visibility_rate": 0.0,
                "is_abnormal_raw": False,
            }

        vis = float(inf.get("visibility_rate", 0.0) or 0.0)
        raw = bool(inf.get("is_abnormal_raw", False)) if inf.get("ok") else False
        if not inf.get("ok"):
            raw = False

        final_status = stab.update(timestamp_sec, raw, vis)
        alert = final_status == STATUS_ABNORMAL

        drawn = draw_pose_overlay(bgr, inf, final_status=final_status, alert_triggered=alert)
        if alert:
            drawn = blend_alert_red(drawn)
        cv2.putText(
            drawn,
            f"t={timestamp_sec:.2f}s",
            (w_i - 160, h_i - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            drawn,
            label,
            (10, h_i - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if drawn.shape[1] != w_i or drawn.shape[0] != h_i:
            drawn = cv2.resize(drawn, (w_i, h_i))

        writer.write(drawn)

        if frame_index in shot_positions and saved_shots < 4:
            shot_path = FIGURES_DIR / f"snap_{stem}_{saved_shots + 1:02d}.jpg"
            enc_ok, buf = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if enc_ok:
                buf.tofile(str(shot_path))
                saved_shots += 1

        frame_index += 1

    cap.release()
    writer.release()
    logger.warning("Демо-видео сохранено: %s (кадров=%d)", out_path.name, frame_index)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    for stem in ("fall4", "normal2", "normal3"):
        render_video_demo(model, stem, label=f"{stem} (final_test)")

    print(f"[OK] Визуализация final_test: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
