"""Extract frames for test2 videos."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

TEST2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TEST2_ROOT.parent

NORMAL_VIDEOS_DIR = PROJECT_ROOT / "data" / "normal"
ARCHIVE_VIDEOS_DIR = PROJECT_ROOT / "data" / "archive_videos"
FRAMES_OUT_DIR = TEST2_ROOT / "data" / "frames"
METADATA_PATH = TEST2_ROOT / "data" / "metadata_test2.json"

FALL_STEMS = {"fall1", "fall2", "fall4"}

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

def collect_video_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    if NORMAL_VIDEOS_DIR.is_dir():
        paths.extend(sorted(p for p in NORMAL_VIDEOS_DIR.iterdir() if p.suffix.lower() == ".mp4"))
    for name in ("fall1.mp4", "fall2.mp4", "fall4.mp4"):
        p = ARCHIVE_VIDEOS_DIR / name
        if p.is_file():
            paths.append(p)
        else:
            logger.warning("Ожидаемое видео не найдено (пропуск): %s", p)
    unique: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

def extract_all_frames() -> tuple[int, int, dict[str, dict[str, float | int | str]]]:
    metadata: dict[str, dict[str, float | int | str]] = {}
    total_written = 0
    videos_ok = 0

    mp4_paths = collect_video_paths()
    if not mp4_paths:
        logger.warning("Нет входных .mp4 (проверьте data/normal/ и data/archive_videos/).")
        return 0, 0, {}

    for video_path in tqdm(mp4_paths, desc="Видео test2", unit="file"):
        key = video_path.name
        stem = video_path.stem

        if key.lower().startswith("fall") and stem.lower() not in FALL_STEMS:
            logger.warning("Архивный ролик не из списка fall1/2/4 (пропуск): %s", key)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Не удалось открыть видео: %s", video_path)
            cap.release()
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            out_path = FRAMES_OUT_DIR / f"{stem}_frame_{frame_idx:05d}.jpg"
            if save_jpg(out_path, frame):
                saved += 1
            frame_idx += 1

        cap.release()

        if frame_idx == 0:
            logger.warning("Видео без кадров: %s", video_path)
            continue

        eff_fps = fps if fps > 0 else 0.0
        duration_sec = round(frame_idx / eff_fps, 1) if eff_fps > 0 else 0.0

        metadata[key] = {
            "fps": round(eff_fps, 4) if eff_fps > 0 else 0.0,
            "total_frames": frame_idx,
            "duration_sec": duration_sec,
            "stem": stem,
        }

        if eff_fps <= 0:
            logger.warning("FPS=0 для %s; total_frames взято по фактическому read (%s)", key, frame_idx)

        if saved != frame_idx:
            logger.warning("Часть кадров не сохранена для %s: %s/%s", key, saved, frame_idx)

        total_written += saved
        videos_ok += 1

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return total_written, videos_ok, metadata

def main() -> None:
    n_frames, n_vid, _ = extract_all_frames()
    print(f"[OK] test2: извлечено {n_frames} кадров из {n_vid} видео. Метаданные: {METADATA_PATH}")

if __name__ == "__main__":
    main()
