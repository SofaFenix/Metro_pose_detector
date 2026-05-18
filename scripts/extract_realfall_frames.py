"""Extract realfall frames; write realfall_meta.json and realfall_gt.csv."""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
_FINAL_TEST_ROOT = PROJECT_ROOT.parent
_REPO_ROOT = _FINAL_TEST_ROOT.parent

VIDEO_REL = Path("data") / "archive_videos" / "realfall.mp4"
VIDEO_PATH = _REPO_ROOT / VIDEO_REL

FRAMES_OUT_DIR = _FINAL_TEST_ROOT / "data" / "frames" / "realfall"
GT_DIR = _FINAL_TEST_ROOT / "data" / "gt"
META_OUT_PATH = GT_DIR / "realfall_meta.json"
GT_CSV_PATH = GT_DIR / "realfall_gt.csv"

INCIDENT_START_SEC = 3.10

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def save_frame_jpeg(path: Path, bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as exc:
        logger.warning("Ошибка сохранения кадра %s: %s", path.name, exc)
        return False

def main() -> None:
    t0 = time.perf_counter()
    if not VIDEO_PATH.is_file():
        logger.warning("Видео не найдено: %s — извлечение пропущено.", VIDEO_PATH)
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        logger.warning("Не удалось открыть видео: %s", VIDEO_PATH)
        return

    fps_prop = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w_prop = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_prop = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_use = fps_prop if fps_prop > 1e-3 else 30.0

    FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        out_path = FRAMES_OUT_DIR / f"realfall_frame_{idx:05d}.jpg"
        if not save_frame_jpeg(out_path, frame):
            logger.warning("Пропуск битого/несохранённого кадра index=%d", idx)
        idx += 1

    cap.release()
    total = idx
    if total == 0:
        logger.warning("Из видео не извлечено ни одного кадра.")
        return

    incident_frame = int(math.ceil(INCIDENT_START_SEC * fps_use - 1e-12))
    incident_frame = max(0, min(incident_frame, total))

    meta = {
        "source_video": str(Path("data") / "archive_videos" / "realfall.mp4"),
        "resolved_path": str(VIDEO_PATH.resolve()),
        "fps": fps_use,
        "total_frames": total,
        "width": w_prop,
        "height": h_prop,
        "opencv_frame_count_hint": n_prop,
        "incident_start_sec": INCIDENT_START_SEC,
        "incident_start_frame_index": incident_frame,
        "extraction_seconds": round(time.perf_counter() - t0, 3),
    }
    META_OUT_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    with GT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video_name", "frame_index", "timestamp_sec", "gt_class"],
        )
        w.writeheader()
        for i in range(total):
            ts = round(i / fps_use, 6) if fps_use > 0 else float(i)
            gtc = "NORMAL" if i < incident_frame else "ABNORMAL_FLOOR_POSTURE"
            w.writerow(
                {
                    "video_name": "realfall.mp4",
                    "frame_index": i,
                    "timestamp_sec": ts,
                    "gt_class": gtc,
                }
            )

    logger.warning(
        "Извлечено кадров=%d, FPS=%.4f, инцидент с frame_index=%d — метаданные: %s",
        total,
        fps_use,
        incident_frame,
        META_OUT_PATH,
    )

if __name__ == "__main__":
    main()
