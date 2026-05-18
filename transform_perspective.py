"""Perspective warp for CCTV angle simulation (scale, narrow, shift_y)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "upfall" / "images_normal"
OUTPUT_DIR = PROJECT_ROOT / "data" / "upfall" / "images_normal_transformed"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCALE_FACTOR = 0.6
NARROWING_FACTOR = 0.7
SHIFT_Y = 0.05

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

def save_jpg(image_path: Path, image: np.ndarray, quality: int = 95) -> bool:
    try:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return False
        buffer.tofile(str(image_path))
        return True
    except Exception:
        return False

def build_points(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    src_points = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ],
        dtype=np.float32,
    )

    w_dst = width * SCALE_FACTOR
    w_top = w_dst * NARROWING_FACTOR
    h_dst = height * SCALE_FACTOR
    cx = width / 2.0
    cy = height / 2.0

    dst_points = np.array(
        [
            [cx - w_top / 2.0, cy - h_dst / 2.0 + height * SHIFT_Y],
            [cx + w_top / 2.0, cy - h_dst / 2.0 + height * SHIFT_Y],
            [cx + w_dst / 2.0, cy + h_dst / 2.0],
            [cx - w_dst / 2.0, cy + h_dst / 2.0],
        ],
        dtype=np.float32,
    )
    return src_points, dst_points

def transform_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    src_points, dst_points = build_points(width, height)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (width, height))

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(INPUT_DIR)
    if not image_paths:
        print(f"[WARN] Нет входных изображений: {INPUT_DIR}")
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

    processed = 0
    total = len(image_paths)

    for image_path in tqdm(readable_paths, desc="Perspective transform"):
        try:
            image = load_image(image_path)
            if image is None or image.size == 0:
                raise ValueError("битое или пустое изображение")

            transformed = transform_image(image)
            output_path = OUTPUT_DIR / f"{image_path.stem}.jpg"
            ok = save_jpg(output_path, transformed, quality=95)
            if not ok:
                raise IOError("не удалось сохранить выходной файл")
            processed += 1
        except Exception as exc:
            print(f"[WARN] Пропуск {image_path.name}: {exc}")

    print(f"Обработано {processed}/{total} изображений. Сохранено в {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
