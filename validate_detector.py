"""YOLO pose model comparison: PCK@0.1, VisRate, FPS."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from config import MODEL_VERSIONS

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "data" / "cctv_incident" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "cctv_incident" / "labels"
RESULTS_JSON_PATH = PROJECT_ROOT / "results" / "detector_validation.json"
RESULTS_MD_PATH = PROJECT_ROOT / "results" / "metrics_pose_comparison.md"
FIGURE_BAR_PATH = PROJECT_ROOT / "figures" / "pose_metrics_comparison.png"
FIGURE_EXAMPLES_PATH = PROJECT_ROOT / "figures" / "detector_examples.png"

IOU_MATCH_THRESHOLD = 0.5
PCK_ALPHA = 0.1
KP_CONF_THRESHOLD = 0.5
MIN_VISIBLE_POINTS_FOR_FRAME = 12
TOTAL_KEYPOINTS = 17
WARMUP_FRAMES = 10
MEASURE_FRAMES = 50
REALTIME_FPS_THRESHOLD = 20.0

COCO_SKELETON = [
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

@dataclass
class GTObject:
    class_id: int
    bbox_xywh: np.ndarray
    keypoints_xy: np.ndarray
    keypoints_vis: np.ndarray

@dataclass
class PredObject:
    bbox_xywh: np.ndarray
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray

@dataclass
class FrameSample:
    path: Path
    image: np.ndarray
    gt_objects: List[GTObject]

def xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box_xywh
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

def bbox_iou_xywh(a_xywh: np.ndarray, b_xywh: np.ndarray) -> float:
    a = xywh_to_xyxy(a_xywh)
    b = xywh_to_xyxy(b_xywh)
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0 else 0.0

def parse_gt_labels(label_path: Path, img_w: int, img_h: int) -> List[GTObject]:
    if not label_path.exists():
        return []
    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []

    gt_objects: List[GTObject] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5 + TOTAL_KEYPOINTS * 3:
            continue
        values = np.array(parts, dtype=np.float32)
        class_id = int(values[0])
        cx, cy, w, h = values[1:5]
        bbox_xywh = np.array([cx * img_w, cy * img_h, w * img_w, h * img_h], dtype=np.float32)
        kp_raw = values[5 : 5 + TOTAL_KEYPOINTS * 3].reshape(TOTAL_KEYPOINTS, 3)
        keypoints_xy = np.stack((kp_raw[:, 0] * img_w, kp_raw[:, 1] * img_h), axis=1).astype(np.float32)
        keypoints_vis = kp_raw[:, 2].astype(np.int32)
        gt_objects.append(GTObject(class_id, bbox_xywh, keypoints_xy, keypoints_vis))
    return gt_objects

def extract_predictions(result) -> List[PredObject]:
    if result is None or result.boxes is None or result.keypoints is None:
        return []
    try:
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf_raw = result.keypoints.conf
        kp_conf = kp_conf_raw.cpu().numpy() if kp_conf_raw is not None else np.zeros((len(boxes_xywh), TOTAL_KEYPOINTS))
    except Exception:
        return []

    count = min(len(boxes_xywh), len(kp_xy), len(kp_conf))
    preds: List[PredObject] = []
    for i in range(count):
        preds.append(
            PredObject(
                bbox_xywh=boxes_xywh[i].astype(np.float32),
                keypoints_xy=kp_xy[i].astype(np.float32),
                keypoints_conf=kp_conf[i].astype(np.float32),
            )
        )
    return preds

def preload_dataset() -> List[FrameSample]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not IMAGES_DIR.exists():
        return []
    image_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in extensions])
    frames: List[FrameSample] = []
    broken_files = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            broken_files += 1
            continue
        h, w = image.shape[:2]
        label_path = LABELS_DIR / f"{path.stem}.txt"
        frames.append(FrameSample(path=path, image=image, gt_objects=parse_gt_labels(label_path, w, h)))
    print(f"Загружено {len(frames)} кадров в память")
    if broken_files > 0:
        print(f"[WARN] Пропущено битых изображений: {broken_files}")
    return frames

def run_timed_predict(model: YOLO, frame_samples: Sequence[FrameSample]) -> Tuple[List[object], float, int]:
    results_per_frame: List[object] = []
    measured_total_time = 0.0
    measured_frames = 0
    warmup_limit = min(WARMUP_FRAMES, len(frame_samples))
    measure_limit = min(MEASURE_FRAMES, max(0, len(frame_samples) - warmup_limit))

    for idx, sample in enumerate(tqdm(frame_samples, desc="Inference", leave=False)):
        if idx < warmup_limit:
            _ = model.predict(source=sample.image, verbose=False)
            continue

        use_measure = measured_frames < measure_limit
        if use_measure:
            t0 = time.perf_counter()
            results = model.predict(source=sample.image, verbose=False)
            measured_total_time += time.perf_counter() - t0
            measured_frames += 1
        else:
            results = model.predict(source=sample.image, verbose=False)
        results_per_frame.append(results[0] if results else None)

    return results_per_frame, measured_total_time, measured_frames

def draw_skeleton(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    valid_mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    canvas = image.copy()
    for a, b in COCO_SKELETON:
        if valid_mask[a] and valid_mask[b]:
            pa = tuple(np.round(keypoints_xy[a]).astype(int))
            pb = tuple(np.round(keypoints_xy[b]).astype(int))
            cv2.line(canvas, pa, pb, color, thickness, lineType=cv2.LINE_AA)
    for idx, ok in enumerate(valid_mask):
        if ok:
            pt = tuple(np.round(keypoints_xy[idx]).astype(int))
            cv2.circle(canvas, pt, 3, color, -1, lineType=cv2.LINE_AA)
    return canvas

def evaluate_predictions(
    model_name: str,
    eval_frames: Sequence[FrameSample],
    frame_results: Sequence[object],
    measured_total_time: float,
    measured_frames: int,
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    total_gt_objects = 0
    matched_gt_objects = 0
    total_valid_gt_points = 0
    total_correct_points = 0
    per_kp_total = np.zeros(TOTAL_KEYPOINTS, dtype=np.int64)
    per_kp_correct = np.zeros(TOTAL_KEYPOINTS, dtype=np.int64)
    total_pred_conf_sum = 0.0
    total_pred_conf_count = 0
    frames_with_visibility = 0

    best_sample: Optional[Dict[str, object]] = None
    best_sample_pck = -1.0

    for sample, raw_result in zip(eval_frames, frame_results):
        preds = extract_predictions(raw_result)
        gt_objects = sample.gt_objects
        total_gt_objects += len(gt_objects)

        frame_visible = any(int(np.sum(pred.keypoints_conf >= KP_CONF_THRESHOLD)) >= MIN_VISIBLE_POINTS_FOR_FRAME for pred in preds)
        if frame_visible:
            frames_with_visibility += 1

        for pred in preds:
            total_pred_conf_sum += float(np.sum(pred.keypoints_conf))
            total_pred_conf_count += int(pred.keypoints_conf.size)

        for gt in gt_objects:
            best_iou = 0.0
            best_pred: Optional[PredObject] = None
            for pred in preds:
                iou = bbox_iou_xywh(gt.bbox_xywh, pred.bbox_xywh)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred

            if best_pred is None or best_iou < IOU_MATCH_THRESHOLD:
                continue

            matched_gt_objects += 1
            # PCK@0.1 formula: dist(pred, gt) < 0.1 * diagonal(gt_bbox)
            diag = float(np.hypot(gt.bbox_xywh[2], gt.bbox_xywh[3]))
            if diag <= 0:
                continue

            vis_mask = gt.keypoints_vis == 2
            if not np.any(vis_mask):
                continue

            dists = np.linalg.norm(best_pred.keypoints_xy - gt.keypoints_xy, axis=1)
            correct_mask = (dists < (PCK_ALPHA * diag)) & vis_mask

            valid_points = int(np.sum(vis_mask))
            correct_points = int(np.sum(correct_mask))
            total_valid_gt_points += valid_points
            total_correct_points += correct_points

            for kp_idx in range(TOTAL_KEYPOINTS):
                if vis_mask[kp_idx]:
                    per_kp_total[kp_idx] += 1
                    if correct_mask[kp_idx]:
                        per_kp_correct[kp_idx] += 1

            instance_pck = correct_points / max(valid_points, 1)
            if instance_pck > best_sample_pck:
                best_sample_pck = instance_pck
                best_sample = {
                    "image_path": str(sample.path),
                    "gt_xy": gt.keypoints_xy.tolist(),
                    "gt_vis": gt.keypoints_vis.tolist(),
                    "pred_xy": best_pred.keypoints_xy.tolist(),
                    "pred_conf": best_pred.keypoints_conf.tolist(),
                }

    pck = total_correct_points / max(total_valid_gt_points, 1)
    detection_rate = matched_gt_objects / max(total_gt_objects, 1)
    vis_rate = 100.0 * frames_with_visibility / max(len(eval_frames), 1)
    mean_kp_conf = total_pred_conf_sum / max(total_pred_conf_count, 1)
    per_kp_accuracy = [(float(per_kp_correct[i]) / max(int(per_kp_total[i]), 1)) for i in range(TOTAL_KEYPOINTS)]
    per_kp_avg = float(np.mean(per_kp_accuracy)) if per_kp_accuracy else 0.0
    fps = measured_frames / measured_total_time if measured_total_time > 0 and measured_frames > 0 else 0.0

    metrics: Dict[str, object] = {
        "model": model_name,
        "status": "OK",
        "pck_at_0_1": pck,
        "detection_rate": detection_rate,
        "mean_keypoint_confidence": mean_kp_conf,
        "visibility_rate_percent": vis_rate,
        "fps": fps,
        "measured_frames": measured_frames,
        "measured_inference_time_sec": measured_total_time,
        "matched_gt_objects": matched_gt_objects,
        "total_gt_objects": total_gt_objects,
        "total_valid_gt_keypoints": total_valid_gt_points,
        "per_keypoint_accuracy": per_kp_accuracy,
        "per_kp_avg": per_kp_avg,
    }
    return metrics, best_sample

def save_markdown(metrics_rows: List[Dict[str, object]]) -> None:
    RESULTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Model | PCK@0.1 | Detection Rate | VisRate@0.5 | FPS | Per-KP Avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            "| {model} | {pck:.4f} | {det:.4f} | {vis:.2f}% | {fps:.2f} | {kp:.4f} |".format(
                model=row.get("model", "N/A"),
                pck=float(row.get("pck_at_0_1", 0.0)),
                det=float(row.get("detection_rate", 0.0)),
                vis=float(row.get("visibility_rate_percent", 0.0)),
                fps=float(row.get("fps", 0.0)),
                kp=float(row.get("per_kp_avg", 0.0)),
            )
        )
    RESULTS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

def plot_metrics(metrics_rows: List[Dict[str, object]]) -> None:
    FIGURE_BAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    models = [str(m.get("model", "N/A")) for m in metrics_rows]
    pck = [float(m.get("pck_at_0_1", 0.0)) for m in metrics_rows]
    fps = [float(m.get("fps", 0.0)) for m in metrics_rows]
    x = np.arange(len(models))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(11, 5))
    b1 = ax1.bar(x - width / 2, pck, width=width, color="#1f77b4", label="PCK@0.1")
    ax1.set_ylabel("PCK@0.1", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.0, max(1.0, (max(pck) * 1.2 if pck else 1.0)))

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, fps, width=width, color="#ff7f0e", label="FPS")
    ax2.set_ylabel("FPS", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    ax1.set_title("Pose Models Comparison (PCK@0.1 vs FPS)")
    ax1.legend([b1, b2], ["PCK@0.1", "FPS"], loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_BAR_PATH, dpi=200)
    plt.close(fig)

def make_collage(best_model: str, worst_model: str, examples: Dict[str, Dict[str, object]]) -> None:
    FIGURE_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    images: List[np.ndarray] = []
    for model_name in [best_model, worst_model]:
        sample = examples.get(model_name)
        if not sample:
            continue
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            continue
        gt_xy = np.asarray(sample["gt_xy"], dtype=np.float32)
        gt_vis_mask = np.asarray(sample["gt_vis"], dtype=np.int32) == 2
        pred_xy = np.asarray(sample["pred_xy"], dtype=np.float32)
        pred_mask = np.asarray(sample["pred_conf"], dtype=np.float32) >= KP_CONF_THRESHOLD
        overlay = draw_skeleton(image, gt_xy, gt_vis_mask, color=(0, 255, 0), thickness=3)
        overlay = draw_skeleton(overlay, pred_xy, pred_mask, color=(0, 0, 255), thickness=2)
        cv2.putText(overlay, model_name, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        images.append(overlay)

    if not images:
        blank = np.zeros((360, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "No examples available", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(FIGURE_EXAMPLES_PATH), blank)
        return

    if len(images) == 1:
        images.append(images[0].copy())
    h = min(img.shape[0] for img in images)
    resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in images[:2]]
    collage = np.hstack(resized)
    cv2.imwrite(str(FIGURE_EXAMPLES_PATH), collage)

def choose_best(metrics: Sequence[Dict[str, object]]) -> str:
    if not metrics:
        return "N/A"
    return max(metrics, key=lambda m: float(m.get("pck_at_0_1", 0.0))).get("model", "N/A")  # type: ignore[return-value]

def choose_realtime(metrics: Sequence[Dict[str, object]]) -> str:
    candidates = [m for m in metrics if float(m.get("fps", 0.0)) > REALTIME_FPS_THRESHOLD]
    if not candidates:
        return "N/A"
    return max(candidates, key=lambda m: float(m.get("pck_at_0_1", 0.0))).get("model", "N/A")  # type: ignore[return-value]

def choose_balanced(metrics: Sequence[Dict[str, object]]) -> str:
    if not metrics:
        return "N/A"
    pck = np.array([float(m.get("pck_at_0_1", 0.0)) for m in metrics], dtype=np.float32)
    fps = np.array([float(m.get("fps", 0.0)) for m in metrics], dtype=np.float32)
    pck_n = pck / max(float(pck.max()), 1e-9)
    fps_n = fps / max(float(fps.max()), 1e-9)
    scores = 0.5 * pck_n + 0.5 * fps_n
    return metrics[int(scores.argmax())].get("model", "N/A")  # type: ignore[return-value]

def main() -> None:
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_BAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_samples = preload_dataset()
    if not frame_samples:
        print(f"[WARN] Нет валидных изображений в {IMAGES_DIR}")
        payload = {"status": "WARNING", "reason": "No valid frames", "models": []}
        RESULTS_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        save_markdown([])
        plot_metrics([])
        make_collage("N/A", "N/A", {})
        return

    results_all: List[Dict[str, object]] = []
    examples_by_model: Dict[str, Dict[str, object]] = {}

    for model_path in MODEL_VERSIONS:
        print(f"\n[INFO] Загрузка модели: {model_path}")
        try:
            model = YOLO(model_path, verbose=False)
        except Exception as e:
            print(f"[WARN] LOAD_FAILED for {model_path}: {e}")
            results_all.append(
                {
                    "model": model_path,
                    "status": "LOAD_FAILED",
                    "error": str(e),
                    "pck_at_0_1": 0.0,
                    "detection_rate": 0.0,
                    "visibility_rate_percent": 0.0,
                    "fps": 0.0,
                    "per_kp_avg": 0.0,
                }
            )
            continue

        try:
            frame_results, measured_time, measured_frames = run_timed_predict(model, frame_samples)
            eval_frames = frame_samples[min(WARMUP_FRAMES, len(frame_samples)) :]
            metrics, best_sample = evaluate_predictions(
                model_name=model_path,
                eval_frames=eval_frames,
                frame_results=frame_results,
                measured_total_time=measured_time,
                measured_frames=measured_frames,
            )
            results_all.append(metrics)
            if best_sample is not None:
                examples_by_model[model_path] = best_sample
        except Exception as e:
            print(f"[WARN] EVAL_FAILED for {model_path}: {e}")
            results_all.append(
                {
                    "model": model_path,
                    "status": "EVAL_FAILED",
                    "error": str(e),
                    "pck_at_0_1": 0.0,
                    "detection_rate": 0.0,
                    "visibility_rate_percent": 0.0,
                    "fps": 0.0,
                    "per_kp_avg": 0.0,
                }
            )

    successful = [m for m in results_all if str(m.get("status")) == "OK"]
    best_model = choose_best(successful)
    worst_model = min(successful, key=lambda m: float(m.get("pck_at_0_1", 0.0))).get("model", "N/A") if successful else "N/A"
    realtime_model = choose_realtime(successful)
    balanced_model = choose_balanced(successful)

    payload = {
        "dataset": {
            "images_dir": str(IMAGES_DIR),
            "labels_dir": str(LABELS_DIR),
            "num_frames_loaded": len(frame_samples),
        },
        "parameters": {
            "iou_match_threshold": IOU_MATCH_THRESHOLD,
            "pck_alpha": PCK_ALPHA,
            "keypoint_conf_threshold": KP_CONF_THRESHOLD,
            "min_visible_points_for_frame": MIN_VISIBLE_POINTS_FOR_FRAME,
            "warmup_frames": WARMUP_FRAMES,
            "measure_frames": MEASURE_FRAMES,
        },
        "models": results_all,
        "recommendations": {
            "max_pck_model": best_model,
            "realtime_balance_model": realtime_model,
            "speed_accuracy_balance_model": balanced_model,
        },
    }
    RESULTS_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    save_markdown(results_all)
    plot_metrics(results_all)
    make_collage(str(best_model), str(worst_model), examples_by_model)

    print(f"Максимальная точность (PCK@0.1): {best_model}")
    print(f"Для real-time баланса (>20 FPS): {realtime_model}")
    print(f"Оптимальный баланс скорость/точность: {balanced_model}")
    print(f"[OK] JSON: {RESULTS_JSON_PATH}")
    print(f"[OK] Markdown: {RESULTS_MD_PATH}")
    print(f"[OK] Figure: {FIGURE_BAR_PATH}")
    print(f"[OK] Examples: {FIGURE_EXAMPLES_PATH}")

if __name__ == "__main__":
    main()
