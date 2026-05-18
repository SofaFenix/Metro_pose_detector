from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "upfall" / "images_normal"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main() -> None:
    if not INPUT_DIR.exists():
        print(f"[WARN] Директория не найдена: {INPUT_DIR}")
        return

    image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not image_paths:
        print(f"[WARN] Нет файлов изображений: {INPUT_DIR}")
        return

    ok_imread = 0
    ok_imdecode = 0
    broken = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None and img.size > 0:
            ok_imread += 1
            continue
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size > 0 else None
            if dec is not None and dec.size > 0:
                ok_imdecode += 1
            else:
                broken.append((path, "imdecode failed"))
        except Exception as exc:
            broken.append((path, str(exc)))

    print(f"Всего файлов: {len(image_paths)}")
    print(f"cv2.imread OK: {ok_imread}")
    print(f"Fallback imdecode OK: {ok_imdecode}")
    print(f"Нечитаемых: {len(broken)}")
    for path, reason in broken[:10]:
        print(f"[BROKEN] {path} :: {reason}")

if __name__ == "__main__":
    main()
