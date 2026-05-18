"""Frame-level GT for time_tune videos."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"

METADATA_PATH = T3_ROOT / "gt" / "metadata.json"
OUTPUT_CSV_PATH = T3_ROOT / "gt" / "frame_level_gt.csv"

GT_CLASS_NORMAL = "NORMAL"
GT_CLASS_ABNORMAL = "ABNORMAL_HORIZONTAL_POSTURE"

# Интервалы аномалий (секунды, inclusive по кадрам)
ABNORMAL_INTERVALS_SEC: dict[str, tuple[float, float]] = {
    "abnorm1": (2.10, 6.50),
    "abnorm2": (2.60, 3.40),
    "abnorm3": (1.50, 5.40),
    "abnorm4": (1.60, 6.00),
}

NORMAL_STEMS = {f"norm{i}" for i in range(1, 5)}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def resolve_stem(video_name: str) -> str:
    return Path(video_name).stem.lower()

def is_normal_video(stem: str) -> bool:
    return stem in NORMAL_STEMS or stem.startswith("norm")

def is_abnormal_video(stem: str) -> bool:
    return stem in ABNORMAL_INTERVALS_SEC or stem.startswith("abnorm")

def abnormal_interval_sec(stem: str) -> tuple[float, float] | None:
    key = stem if stem in ABNORMAL_INTERVALS_SEC else None
    if key is None:
        for k in ABNORMAL_INTERVALS_SEC:
            if stem == k or stem.startswith(k):
                key = k
                break
    if key is None:
        return None
    return ABNORMAL_INTERVALS_SEC[key]

def load_metadata() -> dict[str, dict[str, float | int | str]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Отсутствует {METADATA_PATH}; зависимость: test3_temporal_tuning/scripts/extract_frames_t3.py"
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

def generate_frame_level_gt() -> pd.DataFrame:
    metadata = load_metadata()
    rows: list[dict[str, object]] = []
    processed: list[str] = []

    for canon_key in sorted(metadata.keys(), key=lambda x: x.lower()):
        meta = metadata[canon_key]
        fps = float(meta["fps"])
        total_frames = int(meta["total_frames"])
        if fps <= 0 or total_frames <= 0:
            logger.warning("Некорректная метадата для %s", canon_key)
            continue

        stem = resolve_stem(canon_key)

        if is_normal_video(stem):
            for frame_index in range(total_frames):
                timestamp_sec = round(frame_index / fps, 3)
                rows.append(
                    {
                        "video_name": canon_key,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "gt_class": GT_CLASS_NORMAL,
                        "excluded_from_metrics": False,
                    }
                )
            processed.append(canon_key)
            continue

        if is_abnormal_video(stem):
            interval = abnormal_interval_sec(stem)
            if interval is None:
                logger.warning("Нет интервала GT для %s (пропуск)", canon_key)
                continue
            start_sec, end_sec = interval
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            start_frame = max(0, start_frame)
            end_frame = min(total_frames - 1, end_frame)
            if start_frame > end_frame:
                logger.warning("%s: пустой диапазон GT (%s, %s)", canon_key, start_frame, end_frame)
                continue

            for frame_index in range(total_frames):
                timestamp_sec = round(frame_index / fps, 3)
                if start_frame <= frame_index <= end_frame:
                    gt_class = GT_CLASS_ABNORMAL
                else:
                    gt_class = GT_CLASS_NORMAL
                rows.append(
                    {
                        "video_name": canon_key,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "gt_class": gt_class,
                        "excluded_from_metrics": False,
                    }
                )
            processed.append(canon_key)
            continue

        logger.warning("Видео не norm/abnorm time_tune (пропуск): %s", canon_key)

    out_df = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not out_df.empty:
        out_df = out_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
        out_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    class_counts = out_df["gt_class"].value_counts().to_dict() if not out_df.empty else {}
    print(f"Всего строк GT: {len(out_df)}")
    print(f"Распределение классов: {class_counts}")
    print(f"Обработанные видео: {processed}")
    print(f"Файл: {OUTPUT_CSV_PATH}")
    return out_df

def main() -> None:
    generate_frame_level_gt()

if __name__ == "__main__":
    main()
