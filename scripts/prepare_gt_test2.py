"""Build test2 frame-level GT."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

TEST2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TEST2_ROOT.parent
GT_CONFIG_PATH = PROJECT_ROOT / "data" / "archive_gt_config.csv"
METADATA_PATH = TEST2_ROOT / "data" / "metadata_test2.json"
OUTPUT_CSV_PATH = TEST2_ROOT / "data" / "gt" / "frame_level_gt_test2.csv"

ALLOWED_FALL = {"fall1.mp4", "fall2.mp4", "fall4.mp4"}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_metadata() -> dict[str, dict[str, object]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Отсутствует {METADATA_PATH}; зависимость: test2/scripts/extract_frames_test2.py"
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

def parse_end_frame(end_sec_raw: str | float, fps: float, total_frames: int) -> int:
    text = str(end_sec_raw).strip().lower()
    if text == "end":
        return total_frames - 1
    return int(float(end_sec_raw) * fps)

def is_normal_clip(key: str) -> bool:
    stem = Path(key).stem.lower()
    return stem.startswith("normal")

def is_allowed_fall(key: str) -> bool:
    return key.lower() in {k.lower() for k in ALLOWED_FALL}

def rows_for_normal_video(canon_key: str, meta: dict[str, object]) -> list[dict[str, object]]:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    if fps <= 0 or total_frames <= 0:
        return []
    out: list[dict[str, object]] = []
    for frame_index in range(total_frames):
        timestamp_sec = round(frame_index / fps, 3)
        out.append(
            {
                "video_name": canon_key,
                "source_video": canon_key,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "gt_class": "NORMAL",
                "excluded_from_metrics": False,
            }
        )
    return out

def rows_for_fall_from_config(
    canon_key: str,
    meta: dict[str, object],
    start_sec: float,
    end_sec_raw: str | float,
    gt_class: str,
) -> list[dict[str, object]]:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    if fps <= 0 or total_frames <= 0:
        return []

    start_frame = int(start_sec * fps)
    end_frame = parse_end_frame(end_sec_raw, fps, total_frames)
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)
    if start_frame > end_frame:
        logger.warning("%s: пустой интервал GT после обрезки", canon_key)
        return []

    out: list[dict[str, object]] = []
    for frame_index in range(total_frames):
        timestamp_sec = round(frame_index / fps, 3)
        if start_frame <= frame_index <= end_frame:
            cls = gt_class
        else:
            cls = "NORMAL"
        out.append(
            {
                "video_name": canon_key,
                "source_video": canon_key,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "gt_class": cls,
                "excluded_from_metrics": False,
            }
        )
    return out

def prepare_frame_level_gt() -> pd.DataFrame:
    metadata = load_metadata()
    if not GT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Нет конфигурации GT: {GT_CONFIG_PATH}")

    df_cfg = pd.read_csv(GT_CONFIG_PATH, dtype=str)
    df_cfg["start_sec"] = df_cfg["start_sec"].astype(float)

    fall_cfg_by_name: dict[str, tuple[float, str, str]] = {}
    for _, row in df_cfg.iterrows():
        vn = str(row["video_name"]).strip().strip('"')
        if vn.lower() in {k.lower() for k in ALLOWED_FALL}:
            fall_cfg_by_name[vn] = (float(row["start_sec"]), str(row["end_sec"]), str(row["gt_class"]).strip())

    rows: list[dict[str, object]] = []

    for canon_key in sorted(metadata.keys(), key=lambda x: x.lower()):
        meta = metadata[canon_key]
        if is_normal_clip(canon_key):
            rows.extend(rows_for_normal_video(canon_key, meta))
        elif is_allowed_fall(canon_key):
            if canon_key not in fall_cfg_by_name:
                for k in fall_cfg_by_name:
                    if k.lower() == canon_key.lower():
                        cfg_key = k
                        break
                else:
                    logger.warning("Нет строки в archive_gt_config для %s", canon_key)
                    continue
            else:
                cfg_key = canon_key
            start_sec, end_sec_raw, gt_c = fall_cfg_by_name[cfg_key]
            rows.extend(rows_for_fall_from_config(canon_key, meta, start_sec, end_sec_raw, gt_c))
        else:
            logger.warning("Видео не входит в test2 (пропуск): %s", canon_key)

    out_df = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not out_df.empty:
        out_df = out_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
        out_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    print(f"Всего строк GT: {len(out_df)}")
    if not out_df.empty:
        print(f"Распределение классов: {out_df['gt_class'].value_counts().to_dict()}")
    print(f"Файл: {OUTPUT_CSV_PATH}")
    return out_df

def main() -> None:
    prepare_frame_level_gt()

if __name__ == "__main__":
    main()
