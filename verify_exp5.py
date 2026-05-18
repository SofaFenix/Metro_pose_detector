from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = PROJECT_ROOT / "data" / "upfall" / "images_normal"
TRANSFORMED_DIR = PROJECT_ROOT / "data" / "upfall" / "images_normal_transformed"
OUTPUT_FIGURE = PROJECT_ROOT / "figures" / "exp5_perspective_check.png"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_stems(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    items: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            items[path.stem] = path
    return items

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

def choose_pairs() -> list[tuple[Path, Path]]:
    originals = collect_stems(ORIGINAL_DIR)
    transformed = collect_stems(TRANSFORMED_DIR)
    common = sorted(set(originals.keys()) & set(transformed.keys()))
    pairs: list[tuple[Path, Path]] = []
    for stem in common[:3]:
        pairs.append((originals[stem], transformed[stem]))
    return pairs

def render_collage(pairs: list[tuple[Path, Path]]) -> None:
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), dpi=150)
    for row in range(3):
        for col in range(2):
            axes[row, col].axis("off")

    for idx, (orig_path, trans_path) in enumerate(pairs):
        orig_img = load_image(orig_path)
        trans_img = load_image(trans_path)
        if orig_img is None or trans_img is None:
            continue
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        trans_rgb = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)
        axes[idx, 0].imshow(orig_rgb)
        axes[idx, 0].set_title("Original")
        axes[idx, 1].imshow(trans_rgb)
        axes[idx, 1].set_title("Transformed (~30-45° tilt)")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE)
    plt.close(fig)

def inspect_transformed_frames(sample_size: int = 3) -> tuple[str, float, int]:
    paths = sorted([p for p in TRANSFORMED_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]) if TRANSFORMED_DIR.exists() else []
    if not paths:
        return "WARN", 1.0, 1
    chosen = random.sample(paths, min(sample_size, len(paths)))
    problems = 0
    for path in chosen:
        img = load_image(path)
        if img is None:
            print(f"[WARN] Ошибка загрузки: {path}")
            problems += 1
            continue
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        nan_or_inf_ratio = float((~np.isfinite(img)).sum()) / float(img.size)
        if mean_val < 5:
            print(f"[WARN] Полностью чёрный кадр: {path}")
            problems += 1
        elif nan_or_inf_ratio > 0.001:
            print(f"[WARN] NaN/Inf > 0.1%: {path}")
            problems += 1
        elif std_val < 1:
            print(f"[WARN] Подозрение на артефакт (std < 1): {path}")
            problems += 1
    ratio = problems / max(len(chosen), 1)
    status = "WARN" if ratio > 0.1 else "OK"
    return status, ratio, len(chosen)

def main() -> None:
    pairs = choose_pairs()
    if len(pairs) < 3:
        print("[WARN] Найдено менее 3 пар для визуальной проверки.")
    render_collage(pairs)
    status, ratio, checked = inspect_transformed_frames(sample_size=3)
    print(f"[INFO] Проверено transformed-кадров: {checked}, доля проблемных: {ratio:.2%}")
    print(f"Статус: {status}")
    print(f"Коллаж сохранён: {OUTPUT_FIGURE}")

if __name__ == "__main__":
    main()
