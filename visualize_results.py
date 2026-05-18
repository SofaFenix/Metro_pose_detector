"""Visualization from pipeline predictions."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import config
from run_pipeline_inference import TemporalStabilizer, frame_file_path, infer_single_pose, load_image

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV_PATH = PROJECT_ROOT / "data" / "predictions.csv"
ARCHIVE_FRAMES_ROOT = PROJECT_ROOT / "data" / "archive_frames"
ARCHIVE_METADATA_PATH = PROJECT_ROOT / "data" / "archive_metadata.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
DEMO_FALL4_MP4_PATH = PROJECT_ROOT / "demo_fall4.mp4"
DEMO_NOTFALL_CAMERA120_MP4_PATH = PROJECT_ROOT / "demo_notfall_camera120.mp4"
DEMO_NOTFALL2_CAMERA120_MP4_PATH = PROJECT_ROOT / "demo_notfall2_camera120.mp4"

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

def pick_representative_row_positions(pred_df: pd.DataFrame) -> list[int]:
    """Возвращает до 7 целых позиций строк (iloc) в predictions."""
    thresh = float(config.VISIBILITY_THRESHOLD)

    def head_positions(mask: pd.Series, n: int) -> list[int]:
        hit = np.flatnonzero(mask.to_numpy()).tolist()
        return hit[:n]

    picks: list[int] = []
    picks += head_positions(pred_df["gt_class"] == "ABNORMAL_FLOOR_POSTURE", 2)
    picks += head_positions(pred_df["gt_class"] == "NORMAL", 2)
    occ = head_positions(pred_df["visibility_rate"] < thresh, 1)
    picks += occ
    diff_pos = np.flatnonzero(
        pred_df["is_abnormal_raw"].astype(bool).to_numpy()
        != pred_df["alert_triggered"].astype(bool).to_numpy()
    ).tolist()
    if diff_pos:
        picks.append(int(diff_pos[0]))
    elif len(pred_df) > 0:
        picks.append(len(pred_df) // 2)

    seen: set[int] = set()
    uniq: list[int] = []
    for i in picks:
        ii = int(i)
        if 0 <= ii < len(pred_df) and ii not in seen:
            seen.add(ii)
            uniq.append(ii)
        if len(uniq) >= 7:
            break
    return uniq[:7]

def save_keyframe_figures(model: YOLO, pred_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    picks = pick_representative_row_positions(pred_df)
    for k, ix in enumerate(picks):
        row = pred_df.iloc[ix]
        path = frame_file_path(str(row["video_name"]), int(row["frame_index"]))
        bgr = load_image(path)
        if bgr is None:
            logger.warning("Пропуск визуализации: не прочитан %s", path)
            continue
        inf = infer_single_pose(model, bgr)
        drawn = draw_pose_overlay(
            bgr,
            inf,
            final_status=str(row["final_status"]),
            alert_triggered=bool(row["alert_triggered"]),
        )
        out_path = FIGURES_DIR / f"vis_{k+1:02d}.jpg"
        ok, buf = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            buf.tofile(str(out_path))

def save_representative_frames_for_video(
    model: YOLO,
    video_stem: str,
    *,
    prefix: str,
    max_count: int = 6,
) -> None:
    """Сохраняет репрезентативные кадры конкретного ролика в figures/vis_<prefix>_*.jpg."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pairs = sorted_archive_frame_pairs(video_stem)
    if not pairs:
        logger.warning("Нет кадров для vis %s; пропуск.", video_stem)
        return

    indices = np.linspace(0, len(pairs) - 1, num=min(max_count, len(pairs)), dtype=int)
    fps = load_fps_for_stem(video_stem)
    stab = TemporalStabilizer(config.WINDOW_SIZE, config.MIN_ALERT_FRAMES, config.FALLBACK_K)

    selected = set(indices.tolist())
    saved_counter = 0

    for pos, (frame_index, image_path) in enumerate(pairs):
        bgr = load_image(image_path)
        if bgr is None:
            stab.update(False, True)
            continue

        inf = infer_single_pose(model, bgr)
        vis = float(inf["visibility_rate"]) if inf["ok"] else 0.0
        raw = bool(inf["is_abnormal_raw"]) if inf["ok"] else False
        is_occ = vis < float(config.VISIBILITY_THRESHOLD) or not bool(inf["ok"])
        if not inf["ok"]:
            raw = False
        final_status = stab.update(raw, is_occ)
        alert = final_status == "ABNORMAL_FLOOR_POSTURE"

        if pos in selected:
            drawn = draw_pose_overlay(bgr, inf, final_status=final_status, alert_triggered=alert)
            if alert:
                drawn = blend_alert_red(drawn)

            t_sec = round(frame_index / float(fps), 2)
            h, w = drawn.shape[:2]
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
            out_path = FIGURES_DIR / f"vis_{prefix}_{saved_counter+1:02d}.jpg"
            ok, buf = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                buf.tofile(str(out_path))
                saved_counter += 1

    logger.warning("Сохранено vis для %s: %d кадров", video_stem, saved_counter)

def load_fps_for_stem(video_stem: str) -> float:
    """
    FPS из archive_metadata.json по stem (fall4 ↔ fall4.MP4 и т.п.). Fallback 24 fps.
    """
    if not ARCHIVE_METADATA_PATH.exists():
        return 24.0
    import json

    stem_low = Path(video_stem).stem.lower()
    meta = json.loads(ARCHIVE_METADATA_PATH.read_text(encoding="utf-8"))
    for key in meta:
        if Path(key).stem.lower() == stem_low:
            return float(meta[key]["fps"])
    return 24.0

def sorted_archive_frame_pairs(video_stem: str) -> list[tuple[int, Path]]:
    """Все извлечённые кадры video_stem из archive_frames/, сортировка по индексу."""
    import re

    folder = ARCHIVE_FRAMES_ROOT / video_stem
    if not folder.is_dir():
        logger.warning("Папка кадров отсутствует: %s", folder)
        return []
    pat = re.compile(r"^frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    pairs: list[tuple[int, Path]] = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        m = pat.match(path.name)
        if not m:
            continue
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
    """
    Весь ролик по всем кадрам archive_frames/<video_stem>/; bbox + скелет + оверлей при алерте + таймер и статус.
    """
    pairs = sorted_archive_frame_pairs(video_stem)
    if not pairs:
        logger.warning("Нет кадров для demo %s; пропуск.", video_stem)
        return

    fps = load_fps_for_stem(video_stem)
    probe = load_image(pairs[0][1])
    if probe is None:
        logger.warning("Не прочитан первый кадр %s; demo пропущен.", pairs[0][1])
        return
    h, w = probe.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    stab = TemporalStabilizer(config.WINDOW_SIZE, config.MIN_ALERT_FRAMES, config.FALLBACK_K)

    for frame_index, image_path in pairs:
        bgr = load_image(image_path)
        if bgr is None:
            stab.update(False, True)
            continue
        inf = infer_single_pose(model, bgr)
        vis = float(inf["visibility_rate"]) if inf["ok"] else 0.0
        raw = bool(inf["is_abnormal_raw"]) if inf["ok"] else False
        is_occ = vis < float(config.VISIBILITY_THRESHOLD) or not bool(inf["ok"])
        if not inf["ok"]:
            raw = False
        final_status = stab.update(raw, is_occ)
        alert = final_status == "ABNORMAL_FLOOR_POSTURE"

        frame_use = (
            cv2.resize(bgr, (w, h)) if (bgr.shape[1], bgr.shape[0]) != (w, h) else bgr
        )
        drawn = draw_pose_overlay(frame_use, inf, final_status=final_status, alert_triggered=alert)
        if alert:
            drawn = blend_alert_red(drawn)
        t_sec = round(frame_index / float(fps), 2)
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
        writer.write(drawn)

    writer.release()
    logger.warning("Demo сохранено: %s (%d кадров, fps=%.2f)", output_path.name, len(pairs), fps)

def collect_first_frame_paths(limit: int) -> list[Path]:
    paths: list[Path] = []
    for sub in sorted([p for p in ARCHIVE_FRAMES_ROOT.iterdir() if p.is_dir()]):
        for jpg in sorted(sub.glob("*.jpg")):
            paths.append(jpg)
            if len(paths) >= limit:
                return paths
    return paths

def measure_fps_benchmark(model: YOLO) -> float:
    paths = collect_first_frame_paths(60)
    if len(paths) < 60:
        logger.warning("Меньше 60 кадров в archive_frames для бенча; доступно=%d.", len(paths))
    frames = [load_image(p) for p in paths[:60]]
    frames = [f for f in frames if f is not None][:60]

    warmup = min(10, len(frames))
    measure_len = min(50, len(frames) - warmup)
    if measure_len <= 0:
        logger.warning("Недостаточно кадров для FPS; возвращено 0.0.")
        return 0.0

    for img in frames[:warmup]:
        infer_single_pose(model, img)

    started = time.perf_counter()
    for img in frames[warmup : warmup + measure_len]:
        infer_single_pose(model, img)
    elapsed = time.perf_counter() - started
    return float(measure_len / elapsed if elapsed > 1e-9 else 0.0)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS_CSV_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_CSV_PATH)

    pred_df = pd.read_csv(PREDICTIONS_CSV_PATH).reset_index(drop=True)

    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    save_keyframe_figures(model, pred_df)
    save_representative_frames_for_video(model, "notfall_camera120", prefix="notfall_camera120", max_count=6)
    save_representative_frames_for_video(model, "notfall2_camera120", prefix="notfall2_camera120", max_count=6)

    build_full_demo_roll(
        model,
        "fall4",
        DEMO_FALL4_MP4_PATH,
        label="fall4 (ABNORMAL)",
    )
    build_full_demo_roll(
        model,
        "notfall_camera120",
        DEMO_NOTFALL_CAMERA120_MP4_PATH,
        label="notfall_camera120 (NORMAL)",
    )
    build_full_demo_roll(
        model,
        "notfall2_camera120",
        DEMO_NOTFALL2_CAMERA120_MP4_PATH,
        label="notfall2_camera120 (NORMAL)",
    )

    fps = measure_fps_benchmark(model)
    print(f"Real-time Performance: {fps:.2f} FPS")

if __name__ == "__main__":
    main()
