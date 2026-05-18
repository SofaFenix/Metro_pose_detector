"""Build calibration datasets A/B from upfall images."""
from __future__ import annotations

import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

@dataclass(frozen=True)
class SourceSpec:
    name: str
    images_dir: Path
    labels_dir: Path

@dataclass
class SampleRecord:
    source_name: str
    image_path: Path
    label_path: Path
    gt_class: str

def parse_gt_class(label_path: Path) -> str:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return "UNRELIABLE"
    first_token = text.split()[0]
    class_id = int(float(first_token))
    if class_id == 0:
        return "ABNORMAL"
    if class_id == 1:
        return "NORMAL"
    return "UNRELIABLE"

def collect_samples(sources: Iterable[SourceSpec]) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for source in sources:
        for image_path in sorted(source.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = source.labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            records.append(
                SampleRecord(
                    source_name=source.name,
                    image_path=image_path,
                    label_path=label_path,
                    gt_class=parse_gt_class(label_path),
                )
            )
    return records

def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

def print_distribution(records: list[SampleRecord], title: str) -> None:
    class_counter = Counter(record.gt_class for record in records)
    print(f"\n{title}")
    print(f"Total files: {len(records)}")
    print(f"Class distribution: {dict(class_counter)}")

def print_uniformity_by_blocks(records: list[SampleRecord], blocks: int = 10) -> None:
    if not records:
        print("No records to evaluate block uniformity.")
        return
    block_size = max(1, len(records) // blocks)
    print("Block-wise distribution (post-shuffle):")
    for idx in range(0, len(records), block_size):
        block = records[idx : idx + block_size]
        if not block:
            continue
        counter = Counter(item.gt_class for item in block)
        print(f"  block_{idx // block_size + 1}: {dict(counter)}")

def build_dataset(
    dataset_name: str,
    sources: list[SourceSpec],
) -> None:
    dataset_root = DATA_DIR / dataset_name
    images_out = dataset_root / "images"
    labels_out = dataset_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    clear_dir(images_out)
    clear_dir(labels_out)

    records = collect_samples(sources)
    print_distribution(records, title=f"[{dataset_name}] Before shuffle")

    random.seed(SEED)
    shuffled_records = records[:]
    random.shuffle(shuffled_records)

    for index, record in enumerate(shuffled_records, start=1):
        out_stem = f"frame_{index:06d}"
        image_dst = images_out / f"{out_stem}{record.image_path.suffix.lower()}"
        label_dst = labels_out / f"{out_stem}.txt"
        shutil.copy2(record.image_path, image_dst)
        shutil.copy2(record.label_path, label_dst)

    print_distribution(shuffled_records, title=f"[{dataset_name}] After shuffle")
    print_uniformity_by_blocks(shuffled_records)

def main() -> None:
    sources_a = [
        SourceSpec(
            name="cctv_images",
            images_dir=DATA_DIR / "cctv_incident" / "images",
            labels_dir=DATA_DIR / "cctv_incident" / "labels",
        ),
        SourceSpec(
            name="cctv_images_add",
            images_dir=DATA_DIR / "cctv_incident" / "images_add",
            labels_dir=DATA_DIR / "cctv_incident" / "labels_add",
        ),
        SourceSpec(
            name="upfall_normal",
            images_dir=DATA_DIR / "upfall" / "images_normal",
            labels_dir=DATA_DIR / "upfall" / "labels_normal",
        ),
    ]
    sources_b = [
        SourceSpec(
            name="cctv_images",
            images_dir=DATA_DIR / "cctv_incident" / "images",
            labels_dir=DATA_DIR / "cctv_incident" / "labels",
        ),
        SourceSpec(
            name="cctv_images_add",
            images_dir=DATA_DIR / "cctv_incident" / "images_add",
            labels_dir=DATA_DIR / "cctv_incident" / "labels_add",
        ),
        SourceSpec(
            name="upfall_normal_transformed",
            images_dir=DATA_DIR / "upfall" / "images_normal_transformed",
            labels_dir=DATA_DIR / "upfall" / "labels_normal",
        ),
    ]

    build_dataset(dataset_name="dataset_A", sources=sources_a)
    build_dataset(dataset_name="dataset_B", sources=sources_b)
    print("\nDataset preparation complete.")

if __name__ == "__main__":
    main()
