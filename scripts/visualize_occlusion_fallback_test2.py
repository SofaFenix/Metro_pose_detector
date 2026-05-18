"""Occlusion/fallback visualization from predictions_test2.csv."""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST2_ROOT = PROJECT_ROOT / "test2"
SCRIPTS_DIR = TEST2_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

PREDICTIONS_CSV_PATH = TEST2_ROOT / "data" / "gt" / "predictions_test2.csv"
OUTPUT_DIR = TEST2_ROOT / "figures" / "occlusion_cases"
SUMMARY_PNG_PATH = OUTPUT_DIR / "occlusion_summary.png"

COCO_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _load_pipeline_test2():
    path = SCRIPTS_DIR / "pipeline_test2.py"
    spec = importlib.util.spec_from_file_location("pipeline_test2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def bbox_color_bgr(final_status: str) -> tuple[int, int, int]:
    if final_status == "NORMAL":
        return (0, 220, 0)
    if final_status == "ABNORMAL_FLOOR_POSTURE":
        return (0, 0, 255)
    if final_status == "UNRELIABLE":
        return (0, 255, 255)
    return (180, 180, 180)

def draw_red_cross(img: np.ndarray, x: int, y: int, size: int = 8) -> None:
    cv2.line(img, (x - size, y - size), (x + size, y + size), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, (x - size, y + size), (x + size, y - size), (0, 0, 255), 2, cv2.LINE_AA)

def fallback_active_label(
    visibility_rate: float,
    alert_triggered: bool,
    final_status: str,
    vis_thresh: float,
) -> str:
    if (
        visibility_rate < vis_thresh
        and alert_triggered
        and final_status == "ABNORMAL_FLOOR_POSTURE"
    ):
        return "YES"
    return "NO"

def render_frame_overlay(
    bgr: np.ndarray,
    inf: dict[str, object],
    *,
    row: pd.Series,
    vis_thresh: float,
    conf_thresh: float,
) -> np.ndarray:
    out = bgr.copy()
    h, w_img = out.shape[:2]
    final_status = str(row["final_status"])
    vis = float(row["visibility_rate"])
    alert = bool(row["alert_triggered"])
    fb = fallback_active_label(vis, alert, final_status, vis_thresh)
    color = bbox_color_bgr(final_status)

    if isinstance(inf.get("bbox_xyxy"), tuple) and len(inf["bbox_xyxy"]) == 4:  # type: ignore[arg-type]
        x1, y1, x2, y2 = inf["bbox_xyxy"]  # type: ignore[misc]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    kp = inf.get("keypoints_xy")
    kc = inf.get("keypoints_conf")
    if isinstance(kp, np.ndarray) and isinstance(kc, np.ndarray) and kp.shape[0] >= 17 and kc.shape[0] >= 17:
        for a, b in COCO_EDGES:
            ia, ib = int(a), int(b)
            if ia >= len(kc) or ib >= len(kc):
                continue
            if float(kc[ia]) >= conf_thresh and float(kc[ib]) >= conf_thresh:
                pa = tuple(map(int, kp[ia]))
                pb = tuple(map(int, kp[ib]))
                cv2.line(out, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
        for idx in range(min(17, len(kp), len(kc))):
            x, y = int(kp[idx][0]), int(kp[idx][1])
            if float(kc[idx]) < conf_thresh:
                draw_red_cross(out, x, y, size=7)

    lines = [
        f"Status: {final_status}",
        f"VisRate: {vis:.3f}",
        f"Fallback Active: {fb}",
        f"alert_triggered={alert}",
    ]
    y0 = 26
    panel_h = min(110, h - 10)
    cv2.rectangle(out, (5, 5), (min(520, w_img - 5), 5 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(out, (5, 5), (min(520, w_img - 5), 5 + panel_h), color, 2)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (12, y0 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return out

def find_anchor_indices(df: pd.DataFrame, vis_thresh: float) -> dict[str, int | None]:
    """Возвращает первые индексы строк (iloc) для трёх типов сценариев."""
    df = df.reset_index(drop=True)
    n = len(df)
    type1 = type2 = type3 = None

    for i in tqdm(range(n), desc="Поиск сценариев (Type1/2/3)"):
        row = df.iloc[i]
        vis = float(row["visibility_rate"])
        fs = str(row["final_status"])
        alert = bool(row["alert_triggered"])

        if type1 is None and vis < vis_thresh and fs == "UNRELIABLE" and not alert:
            type1 = i

        if type2 is None and vis < vis_thresh and fs == "ABNORMAL_FLOOR_POSTURE" and alert:
            type2 = i

        if type3 is None and i > 0:
            prev = df.iloc[i - 1]
            same_vid = str(row["video_name"]) == str(prev["video_name"])
            if (
                same_vid
                and float(prev["visibility_rate"]) < vis_thresh
                and vis >= vis_thresh
            ):
                type3 = i

        if type1 is not None and type2 is not None and type3 is not None:
            break

    return {"type1_unreliable": type1, "type2_fallback_alert": type2, "type3_recovery": type3}

def collect_series_indices(df: pd.DataFrame, anchor: int, *, span: int = 2) -> list[int]:
    """Серия из до (2*span+1) кадров вокруг anchor, в границах одного видео."""
    if anchor is None or anchor < 0:
        return []
    vid = str(df.iloc[anchor]["video_name"])
    lo = anchor
    hi = anchor
    for _ in range(span):
        if lo - 1 >= 0 and str(df.iloc[lo - 1]["video_name"]) == vid:
            lo -= 1
        else:
            break
    for _ in range(span):
        if hi + 1 < len(df) and str(df.iloc[hi + 1]["video_name"]) == vid:
            hi += 1
        else:
            break
    return list(range(lo, hi + 1))

def save_series(
    df: pd.DataFrame,
    indices: list[int],
    case_prefix: str,
    model: YOLO,
    pipeline_mod,
    vis_thresh: float,
    conf_thresh: float,
) -> list[np.ndarray]:
    """Сохраняет JPG; возвращает список RGB-кадров для коллажа (успешные)."""
    thumbs: list[np.ndarray] = []
    for j, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        video_name = str(row["video_name"])
        frame_index = int(row["frame_index"])
        path = pipeline_mod.frame_file_path(video_name, frame_index)
        bgr = pipeline_mod.load_image(path)
        if bgr is None:
            logger.warning("Пропуск кадра (файл отсутствует или битый): %s", path)
            continue
        try:
            inf = pipeline_mod.infer_single_pose(model, bgr)
        except Exception as exc:
            logger.warning("YOLO inference failed для %s: %s", path, exc)
            continue
        try:
            drawn = render_frame_overlay(
                bgr,
                inf,
                row=row,
                vis_thresh=vis_thresh,
                conf_thresh=conf_thresh,
            )
        except Exception as exc:
            logger.warning("Оверлей не построен для %s: %s", path, exc)
            continue
        out_path = OUTPUT_DIR / f"{case_prefix}_frame_{j:02d}.jpg"
        ok, buf = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            buf.tofile(str(out_path))
            rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
            thumbs.append(rgb)
        else:
            logger.warning("Не удалось закодировать JPG: %s", out_path)
    return thumbs

def build_summary_collage(case_thumbs: dict[str, list[np.ndarray]], titles: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    order = ["type1_unreliable", "type2_fallback_alert", "type3_recovery"]
    for ax, key in zip(axes, order):
        imgs = case_thumbs.get(key, [])
        if imgs:
            ax.imshow(imgs[len(imgs) // 2])
        else:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", fontsize=14)
        ax.set_title(titles.get(key, key), fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(SUMMARY_PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Коллаж сохранён: %s", SUMMARY_PNG_PATH)

def main() -> None:
    vis_thresh = float(config.VISIBILITY_THRESHOLD)
    conf_thresh = float(config.CONFIDENCE_THRESHOLD)

    if not PREDICTIONS_CSV_PATH.is_file():
        raise FileNotFoundError(PREDICTIONS_CSV_PATH)

    df = pd.read_csv(PREDICTIONS_CSV_PATH)
    required = {
        "video_name",
        "frame_index",
        "visibility_rate",
        "final_status",
        "alert_triggered",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {missing}")

    df = df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)

    anchors = find_anchor_indices(df, vis_thresh)
    logger.info("Якоря сценариев (iloc): %s", anchors)

    pipeline_mod = _load_pipeline_test2()
    model = YOLO(config.DEFAULT_MODEL, verbose=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_thumbs: dict[str, list[np.ndarray]] = {}
    titles = {
        "type1_unreliable": "Type 1: UNRELIABLE (окклюзия, alert=False)",
        "type2_fallback_alert": "Type 2: Fallback удерживает ABNORMAL + alert",
        "type3_recovery": "Type 3: Восстановление (VisRate ≥ порога)",
    }

    mapping = [
        ("type1_unreliable", "case1_unreliable"),
        ("type2_fallback_alert", "case2_fallback_alert"),
        ("type3_recovery", "case3_recovery"),
    ]

    for key, prefix in mapping:
        anchor = anchors.get(key)
        anchor_i = anchor if isinstance(anchor, int) else None
        idxs = collect_series_indices(df, anchor_i, span=2) if anchor_i is not None else []
        if len(idxs) > 5:
            idxs = idxs[:5]
        if not idxs:
            logger.warning("Сценарий %s не найден или пустая серия.", key)
            case_thumbs[key] = []
            continue
        case_thumbs[key] = save_series(
            df,
            idxs,
            prefix,
            model,
            pipeline_mod,
            vis_thresh,
            conf_thresh,
        )

    build_summary_collage(case_thumbs, titles)
    print(f"[OK] Артефакты: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
