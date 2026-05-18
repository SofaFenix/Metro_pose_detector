"""frame_level_gt.csv from archive_gt_config.csv and archive_metadata.json."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

GT_CONFIG_PATH = PROJECT_ROOT / "data" / "archive_gt_config.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "archive_metadata.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "archive_annotations" / "frame_level_gt.csv"

# GT aggregate name maps to notfall1..notfall5 in metadata
NOTFALL_BATCH_MARKER = "not_fall(1-5, все ролики).mp4"

DEFAULT_GT_CONFIG_CSV = """video_name,start_sec,end_sec,gt_class
"not_fall(1-5, все ролики).mp4",0.0,end,NORMAL
fall1.mp4,0.0,end,ABNORMAL_FLOOR_POSTURE
fall2.mp4,3.40,end,ABNORMAL_FLOOR_POSTURE
fall3.mp4,4.00,end,ABNORMAL_FLOOR_POSTURE
fall4.mp4,3.70,end,ABNORMAL_FLOOR_POSTURE
"""

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def ensure_gt_config_template() -> None:
    """Write archive_gt_config.csv template if missing."""
    GT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GT_CONFIG_PATH.exists():
        GT_CONFIG_PATH.write_text(DEFAULT_GT_CONFIG_CSV.strip() + "\n", encoding="utf-8")

def load_metadata() -> dict[str, dict[str, float | int]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Метаданные не найдены: {METADATA_PATH}; зависимость: extract_archive_frames.py."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

def parse_end_frame(end_sec_raw: str | float, fps: float, total_frames: int) -> int:
    text = str(end_sec_raw).strip().lower()
    if text == "end":
        return total_frames - 1
    # end_frame = int(end_sec * fps)
    return int(float(end_sec_raw) * fps)

def resolve_metadata_video_keys(video_name_raw: str, metadata: dict[str, object]) -> list[str]:
    """Map CSV video_name to metadata keys."""
    vn = video_name_raw.strip().strip('"')
    if vn in metadata:
        return [vn]
    lowered = vn.lower()
    for key in metadata:
        if key.lower() == lowered:
            return [key]
    if vn == NOTFALL_BATCH_MARKER:
        nf = sorted(
            k
            for k in metadata.keys()
            if Path(k).stem.lower().startswith("notfall") and Path(k).suffix.lower() == ".mp4"
        )
        return nf
    return []

def prepare_frame_level_gt() -> pd.DataFrame:
    ensure_gt_config_template()
    metadata = load_metadata()
    df_cfg = pd.read_csv(GT_CONFIG_PATH, dtype=str)
    # Числовые старты читаем как float
    df_cfg["start_sec"] = df_cfg["start_sec"].astype(float)

    rows: list[dict[str, object]] = []
    processed_videos: list[str] = []

    for _, row in df_cfg.iterrows():
        video_name = str(row["video_name"]).strip()
        keys = resolve_metadata_video_keys(video_name, metadata)
        if not keys:
            logger.warning("Видео отсутствует в метаданных (пропуск строки): %s", video_name)
            continue

        for canon_key in keys:
            meta = metadata[canon_key]
            fps = float(meta["fps"])
            total_frames = int(meta["total_frames"])
            if fps <= 0 or total_frames <= 0:
                logger.warning("Некорректная метадата для %s", canon_key)
                continue

            start_sec = float(row["start_sec"])
            # start_frame = int(start_sec * fps)
            start_frame = int(start_sec * fps)
            end_frame = parse_end_frame(row["end_sec"], fps, total_frames)
            gt_class = str(row["gt_class"]).strip()

            if start_frame < 0:
                logger.warning("%s: start_frame < 0, обрезано до 0", canon_key)
                start_frame = 0

            max_idx = total_frames - 1
            if end_frame > max_idx:
                logger.warning(
                    "%s: end_frame=%s выходит за границы (total_frames=%s), обрезано до %s",
                    canon_key,
                    end_frame,
                    total_frames,
                    max_idx,
                )
                end_frame = max_idx

            if start_frame > end_frame:
                logger.warning(
                    "%s: пустой диапазон после обрезки (start=%s, end=%s)", canon_key, start_frame, end_frame
                )
                continue

            processed_videos.append(canon_key)
            for frame_index in range(start_frame, end_frame + 1):
                timestamp_sec = round(frame_index / fps, 3)
                rows.append(
                    {
                        "video_name": canon_key,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "gt_class": gt_class,
                        "excluded_from_metrics": False,
                    }
                )

    out_df = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not out_df.empty:
        out_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    # Сводка
    class_counts = out_df["gt_class"].value_counts().to_dict() if not out_df.empty else {}
    print(f"Всего кадров размечено: {len(out_df)}")
    print(f"Распределение по классам: {class_counts}")
    print(f"Обработанные видео (по строкам конфига): {processed_videos}")
    return out_df

def main() -> None:
    prepare_frame_level_gt()

if __name__ == "__main__":
    main()
