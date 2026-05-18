"""test2 demo: fall4, normal2, normal3."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TEST2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TEST2_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

from pipeline_test2 import TemporalMsStabilizer, WINDOW_MS, infer_single_pose, load_image  # noqa: E402

METADATA_PATH = TEST2_ROOT / "data" / "metadata_test2.json"
FIGURES_DIR = TEST2_ROOT / "figures"

TARGET_STEMS = ("fall4", "normal2", "normal3")

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

def status_color(status: str) -> tuple[int, int, int]:
    if status == "NORMAL":
        return (0, 200, 0)
    if status == "ABNORMAL_FLOOR_POSTURE":
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

    if isinstance(inf["bbox_xyxy"], tuple) and len(inf["bbox_xyxy"]) == 4:
        x1, y1, x2, y2 = inf["bbox_xyxy"]  # type: ignore[misc]
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

def load_fps_for_stem(video_stem: str) -> float:
    if not METADATA_PATH.exists():
        return 24.0
    stem_low = Path(video_stem).stem.lower()
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    for key in meta:
        entry = meta[key]
        if str(entry.get("stem", Path(key).stem)).lower() == stem_low or Path(key).stem.lower() == stem_low:
            return float(entry["fps"]) if float(entry.get("fps", 0) or 0) > 0 else 24.0
    return 24.0

def resolve_video_name_for_stem(video_stem: str) -> str:
    if not METADATA_PATH.exists():
        return f"{video_stem}.mp4"
    stem_low = video_stem.lower()
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    for key in meta:
        if Path(key).stem.lower() == stem_low:
            return key
    return f"{video_stem}.mp4"

def sorted_test2_frame_pairs(video_stem: str) -> list[tuple[int, Path]]:
    pat = re.compile(rf"^{re.escape(video_stem)}_frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    folder = TEST2_ROOT / "data" / "frames"
    if not folder.is_dir():
        return []
    pairs: list[tuple[int, Path]] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        m = pat.match(path.name)
        if m:
            pairs.append((int(m.group(1)), path))
    pairs.sort(key=lambda x: x[0])
    return pairs

def build_full_demo_roll(
    model: YOLO,
    video_stem: str,
    output_path: Path,
    *,
    label: str,
) -> None:
    pairs = sorted_test2_frame_pairs(video_stem)
    if not pairs:
        logger.warning("Нет кадров для demo %s", video_stem)
        return

    fps = load_fps_for_stem(video_stem)
    probe = load_image(pairs[0][1])
    if probe is None:
        logger.warning("Не прочитан первый кадр %s", pairs[0][1])
        return
    h, w = probe.shape[:2]

    video_key = resolve_video_name_for_stem(video_stem)
    stab = TemporalMsStabilizer(window_ms=WINDOW_MS, fallback_k=int(config.FALLBACK_K))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))

    for frame_index, image_path in pairs:
        bgr = load_image(image_path)
        timestamp_sec = round(frame_index / float(fps), 6) if fps > 0 else float(frame_index)

        if bgr is None:
            stab.update(timestamp_sec, False, True)
            continue

        inf = infer_single_pose(model, bgr)
        vis = float(inf["visibility_rate"]) if inf["ok"] else 0.0
        raw = bool(inf["is_abnormal_raw"]) if inf["ok"] else False
        is_occ = vis < float(config.VISIBILITY_THRESHOLD) or not bool(inf["ok"])
        if not inf["ok"]:
            raw = False

        final_status = stab.update(timestamp_sec, raw, is_occ)
        alert = final_status == "ABNORMAL_FLOOR_POSTURE"

        frame_use = cv2.resize(bgr, (w, h)) if (bgr.shape[1], bgr.shape[0]) != (w, h) else bgr
        drawn = draw_pose_overlay(frame_use, inf, final_status=final_status, alert_triggered=alert)
        if alert:
            drawn = blend_alert_red(drawn)

        t_sec = round(frame_index / float(fps), 2) if fps > 0 else float(frame_index)
        cv2.putText(
            drawn,
            f"t={t_sec:.2f}s",
            (w - 160, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            drawn,
            f"clip={label}",
            (10, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            drawn,
            f"vid={video_key}",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        writer.write(drawn)

    writer.release()
    logger.warning("Demo test2 сохранено: %s (%d кадров)", output_path.name, len(pairs))

def save_sample_frames(
    model: YOLO,
    video_stem: str,
    *,
    prefix: str,
    max_count: int = 6,
) -> None:
    pairs = sorted_test2_frame_pairs(video_stem)
    if not pairs:
        logger.warning("Нет кадров для vis %s", video_stem)
        return

    fps = load_fps_for_stem(video_stem)
    stab = TemporalMsStabilizer(window_ms=WINDOW_MS, fallback_k=int(config.FALLBACK_K))
    indices = np.linspace(0, len(pairs) - 1, num=min(max_count, len(pairs)), dtype=int)
    selected = set(indices.tolist())
    saved_counter = 0

    for pos, (frame_index, image_path) in enumerate(pairs):
        timestamp_sec = round(frame_index / float(fps), 6) if fps > 0 else float(frame_index)
        bgr = load_image(image_path)
        if bgr is None:
            stab.update(timestamp_sec, False, True)
            continue

        inf = infer_single_pose(model, bgr)
        vis = float(inf["visibility_rate"]) if inf["ok"] else 0.0
        raw = bool(inf["is_abnormal_raw"]) if inf["ok"] else False
        is_occ = vis < float(config.VISIBILITY_THRESHOLD) or not bool(inf["ok"])
        if not inf["ok"]:
            raw = False
        final_status = stab.update(timestamp_sec, raw, is_occ)
        alert = final_status == "ABNORMAL_FLOOR_POSTURE"

        if pos in selected:
            drawn = draw_pose_overlay(bgr, inf, final_status=final_status, alert_triggered=alert)
            if alert:
                drawn = blend_alert_red(drawn)
            h, w = drawn.shape[:2]
            cv2.putText(
                drawn,
                f"t={timestamp_sec:.2f}s",
                (w - 160, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out_path = FIGURES_DIR / f"vis_{prefix}_{saved_counter + 1:02d}.jpg"
            ok, buf = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                buf.tofile(str(out_path))
                saved_counter += 1

    logger.warning("Сохранено vis для %s: %d кадров", video_stem, saved_counter)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    for stem in TARGET_STEMS:
        save_sample_frames(model, stem, prefix=stem, max_count=6)
        demo_name = f"demo_{stem}.mp4"
        build_full_demo_roll(
            model,
            stem,
            FIGURES_DIR / demo_name,
            label=f"{stem} (test2)",
        )

    print(f"[OK] test2 визуализация: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
