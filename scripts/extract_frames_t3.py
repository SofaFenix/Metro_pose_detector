"""Extract frames from data/time_tune/."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"

TIME_TUNE_VIDEOS_DIR = PROJECT_ROOT / "data" / "time_tune"
FRAMES_ROOT = T3_ROOT / "frames"
METADATA_PATH = T3_ROOT / "gt" / "metadata.json"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def save_jpg(image_path: Path, image: np.ndarray, quality: int = 95) -> bool:
    try:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return False
        image_path.parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(str(image_path))
        return True
    except Exception as exc:
        logger.warning("Не удалось сохранить кадр %s: %s", image_path, exc)
        return False

def list_mp4_files(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        logger.warning("Каталог видео не найден: %s", videos_dir)
        return []
    return sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")

def extract_all_frames() -> tuple[int, int, list[str], dict[str, dict[str, float | int | str]]]:
    metadata: dict[str, dict[str, float | int | str]] = {}
    total_frames_written = 0
    videos_ok = 0
    processed: list[str] = []

    mp4_paths = list_mp4_files(TIME_TUNE_VIDEOS_DIR)
    if not mp4_paths:
        logger.warning("Нет .mp4 в %s", TIME_TUNE_VIDEOS_DIR)

    for video_path in tqdm(mp4_paths, desc="time_tune videos", unit="file"):
        key = video_path.name
        stem = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Не удалось открыть видео (пропуск): %s", video_path)
            cap.release()
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        out_dir = FRAMES_ROOT / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
            if save_jpg(out_path, frame):
                saved += 1
            frame_idx += 1

        cap.release()

        if frame_idx == 0:
            logger.warning("Видео без кадров (пропуск): %s", video_path)
            continue

        duration_sec = round(frame_idx / fps, 3) if fps > 0 else 0.0
        metadata[key] = {
            "fps": round(fps, 4) if fps > 0 else 0.0,
            "total_frames": frame_idx,
            "duration_sec": duration_sec,
            "stem": stem,
        }

        if saved != frame_idx:
            logger.warning("Часть кадров не сохранена для %s: %s/%s", key, saved, frame_idx)

        total_frames_written += saved
        videos_ok += 1
        processed.append(key)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return total_frames_written, videos_ok, processed, metadata

def main() -> None:
    total_frames, video_count, processed, _ = extract_all_frames()
    print(f"[OK] test3: извлечено {total_frames} кадров из {video_count} видео.")
    print(f"Метаданные: {METADATA_PATH}")
    print(f"Обработанные видео: {processed}")

if __name__ == "__main__":
    main()
