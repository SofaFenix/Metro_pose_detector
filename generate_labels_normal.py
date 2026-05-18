from __future__ import annotations

# YOLO Pose labels vis=0; inference uses conf>=0.5

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "upfall" / "images_normal"
OUTPUT_DIR = PROJECT_ROOT / "data" / "upfall" / "labels_normal"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_yolo_pose_line() -> str:
    class_id = "1"
    bbox = ["0.5", "0.5", "0.9", "0.9"]
    # 17 keypoints * (x, y, vis)
    keypoints = ["0.0", "0.0", "0.0"] * 17
    values = [class_id, *bbox, *keypoints]
    return " ".join(values)

def collect_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

def load_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is not None and image.size > 0:
        return image
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            return None
        return decoded
    except Exception:
        return None

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(INPUT_DIR)
    if not image_paths:
        print(f"[WARN] Нет изображений для разметки: {INPUT_DIR}")
        return

    readable_paths: list[Path] = []
    unreadable_paths: list[Path] = []
    for image_path in image_paths:
        if load_image(image_path) is None:
            unreadable_paths.append(image_path)
        else:
            readable_paths.append(image_path)
    print(f"[INFO] Preflight: readable={len(readable_paths)}, unreadable={len(unreadable_paths)}")
    for bad_path in unreadable_paths[:5]:
        print(f"[WARN] Нечитаемый файл: {bad_path}")

    yolo_line = build_yolo_pose_line()
    processed = 0
    skipped = 0

    for image_path in tqdm(readable_paths, desc="Generating normal labels"):
        try:
            image = load_image(image_path)
            if image is None or image.size == 0:
                raise ValueError("битое или пустое изображение")
            output_label = OUTPUT_DIR / f"{image_path.stem}.txt"
            output_label.write_text(yolo_line + "\n", encoding="utf-8")
            processed += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Пропуск {image_path.name}: {exc}")

    print(f"Сгенерировано лейблов: {processed}/{len(image_paths)}. Пропущено: {skipped + len(unreadable_paths)}.")
    print(f"Сохранено в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
