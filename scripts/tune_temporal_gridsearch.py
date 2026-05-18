"""Grid search WINDOW_MS x MIN_ALERT_RATIO."""
from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"
SCRIPTS_DIR = T3_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402

from pipeline_t3_gridsearch import (  # noqa: E402
    GT_CSV_PATH,
    STATUS_ABNORMAL,
    apply_temporal_stabilizer,
    run_raw_feature_extraction,
)

RESULTS_CSV_PATH = T3_ROOT / "results" / "grid_search_time_tune.csv"
PARETO_PNG_PATH = T3_ROOT / "figures" / "pareto_frontier_time_tune.png"

WINDOW_MS_GRID = [300, 400, 500, 600, 700, 800]
MIN_ALERT_RATIO_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def extract_contiguous_intervals(
    df: pd.DataFrame,
    *,
    video_col: str = "video_name",
    index_col: str = "frame_index",
    time_col: str = "timestamp_sec",
    active_col: str,
) -> list[dict[str, object]]:
    """Список интервалов {video_name, start_idx, end_idx, start_sec, end_sec}."""
    intervals: list[dict[str, object]] = []
    for video_name, grp in df.groupby(video_col, sort=False):
        grp = grp.sort_values(index_col, kind="mergesort").reset_index(drop=True)
        flags = grp[active_col].astype(bool).to_numpy()
        in_run = False
        start_i = 0
        for i, active in enumerate(flags):
            if active and not in_run:
                in_run = True
                start_i = i
            elif not active and in_run:
                intervals.append(
                    {
                        "video_name": video_name,
                        "start_idx": int(grp.iloc[start_i][index_col]),
                        "end_idx": int(grp.iloc[i - 1][index_col]),
                        "start_sec": float(grp.iloc[start_i][time_col]),
                        "end_sec": float(grp.iloc[i - 1][time_col]),
                    }
                )
                in_run = False
        if in_run:
            intervals.append(
                {
                    "video_name": video_name,
                    "start_idx": int(grp.iloc[start_i][index_col]),
                    "end_idx": int(grp.iloc[-1][index_col]),
                    "start_sec": float(grp.iloc[start_i][time_col]),
                    "end_sec": float(grp.iloc[-1][time_col]),
                }
            )
    return intervals

def _gt_active_mask(gt_df: pd.DataFrame) -> pd.DataFrame:
    out = gt_df.copy()
    out["_active"] = out["gt_class"] == STATUS_ABNORMAL
    return out

def gt_anomaly_intervals(gt_df: pd.DataFrame) -> list[dict[str, object]]:
    tmp = _gt_active_mask(gt_df)
    return extract_contiguous_intervals(tmp, active_col="_active")

def alert_intervals(pred_df: pd.DataFrame) -> list[dict[str, object]]:
    tmp = pred_df.copy()
    if "timestamp_sec" not in tmp.columns:
        tmp["timestamp_sec"] = tmp["timestamp_ms"] / 1000.0
    tmp["_active"] = tmp["alert_triggered"].astype(bool)
    return extract_contiguous_intervals(tmp, active_col="_active")

def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return not (a_end < b_start or b_end < a_start)

def compute_event_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, float]:
    """
    Event_Recall, Event_Precision, F1_event, Latency (мс), FAR (алертов/мин на норме).
    """
    gt_df = gt_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)
    pred_df = pred_df.sort_values(["video_name", "frame_index"], kind="mergesort").reset_index(drop=True)

    if "timestamp_sec" not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["timestamp_sec"] = pred_df["timestamp_ms"] / 1000.0

    gt_intervals = gt_anomaly_intervals(gt_df)
    alert_runs = alert_intervals(pred_df)

    # Event recall
    detected = 0
    latencies_ms: list[float] = []

    pred_alerts = pred_df.loc[pred_df["alert_triggered"].astype(bool)].copy()

    for inc in gt_intervals:
        vid = inc["video_name"]
        s_sec, e_sec = float(inc["start_sec"]), float(inc["end_sec"])
        mask = (
            (pred_alerts["video_name"] == vid)
            & (pred_alerts["timestamp_sec"] >= s_sec)
            & (pred_alerts["timestamp_sec"] <= e_sec)
        )
        hits = pred_alerts.loc[mask]
        if len(hits) > 0:
            detected += 1
            first_alert_t = float(hits["timestamp_sec"].min())
            latencies_ms.append(max(0.0, (first_alert_t - s_sec) * 1000.0))

    n_incidents = len(gt_intervals)
    event_recall = detected / n_incidents if n_incidents > 0 else 0.0

    # Event precision
    gt_by_video: dict[str, list[tuple[float, float]]] = {}
    for inc in gt_intervals:
        gt_by_video.setdefault(inc["video_name"], []).append((float(inc["start_sec"]), float(inc["end_sec"])))

    tp_alerts = 0
    for ar in alert_runs:
        vid = ar["video_name"]
        a0, a1 = float(ar["start_sec"]), float(ar["end_sec"])
        matched = False
        for g0, g1 in gt_by_video.get(vid, []):
            if _interval_overlap(a0, a1, g0, g1):
                matched = True
                break
        if matched:
            tp_alerts += 1

    n_alert_events = len(alert_runs)
    event_precision = tp_alerts / n_alert_events if n_alert_events > 0 else 0.0

    if event_precision + event_recall > 0:
        f1_event = 2 * event_precision * event_recall / (event_precision + event_recall)
    else:
        f1_event = 0.0

    latency_ms = float(np.mean(latencies_ms)) if latencies_ms else float("nan")

    # FAR (alerts/min)
    is_normal_video = gt_df.groupby("video_name")["gt_class"].apply(
        lambda s: (s == STATUS_ABNORMAL).sum() == 0
    )
    normal_videos = is_normal_video[is_normal_video].index.tolist()

    false_alerts = 0
    normal_duration_sec = 0.0
    for vid in normal_videos:
        g = gt_df.loc[gt_df["video_name"] == vid]
        if g.empty:
            continue
        normal_duration_sec += float(g["timestamp_sec"].max()) - float(g["timestamp_sec"].min()) + (
            1.0 / 30.0
        )
        p = pred_df.loc[(pred_df["video_name"] == vid) & pred_df["alert_triggered"].astype(bool)]
        false_alerts += len(p)

    normal_minutes = normal_duration_sec / 60.0 if normal_duration_sec > 0 else 0.0
    far = false_alerts / normal_minutes if normal_minutes > 0 else 0.0

    return {
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "f1_event": float(f1_event),
        "latency_ms": float(latency_ms),
        "far_alerts_per_min": float(far),
        "n_gt_incidents": int(n_incidents),
        "n_detected_incidents": int(detected),
        "n_alert_events": int(n_alert_events),
        "false_alerts_normal": int(false_alerts),
    }

def pareto_frontier_mask(df: pd.DataFrame, *, x_col: str, y_col: str) -> np.ndarray:
    """
    Недоминируемые точки: максимизируем F1 (y), минимизируем Latency (x).
  """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            is_pareto[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            if (y[j] >= y[i] and x[j] <= x[i]) and (y[j] > y[i] or x[j] < x[i]):
                is_pareto[i] = False
                break
    return is_pareto

def run_grid_search() -> pd.DataFrame:
    gt_df = pd.read_csv(GT_CSV_PATH)
    raw_df = run_raw_feature_extraction(gt_df, use_cache=True)

    combos = list(itertools.product(WINDOW_MS_GRID, MIN_ALERT_RATIO_GRID))
    rows: list[dict[str, object]] = []

    for window_ms, min_ratio in tqdm(combos, desc="Grid search temporal"):
        pred_df = apply_temporal_stabilizer(
            raw_df,
            window_ms=float(window_ms),
            min_alert_ratio=float(min_ratio),
            fallback_k=int(config.FALLBACK_K),
        )
        metrics = compute_event_metrics(pred_df, gt_df)
        rows.append(
            {
                "window_ms": int(window_ms),
                "min_alert_ratio": float(min_ratio),
                "fallback_k": int(config.FALLBACK_K),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    results["is_pareto"] = pareto_frontier_mask(
        results, x_col="latency_ms", y_col="f1_event"
    )

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")

    _plot_pareto(results)
    return results

def _plot_pareto(results: pd.DataFrame) -> None:
    PARETO_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_df = results[np.isfinite(results["latency_ms"])].copy()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="latency_ms",
        y="f1_event",
        hue="is_pareto",
        palette={True: "crimson", False: "steelblue"},
        s=80,
        alpha=0.85,
    )

    for _, r in plot_df.iterrows():
        plt.annotate(
            f"{int(r['window_ms'])}/{r['min_alert_ratio']:.1f}",
            (r["latency_ms"], r["f1_event"]),
            fontsize=7,
            alpha=0.75,
            xytext=(3, 3),
            textcoords="offset points",
        )

    pareto_pts = plot_df.loc[plot_df["is_pareto"]].sort_values("latency_ms")
    if len(pareto_pts) >= 2:
        plt.plot(
            pareto_pts["latency_ms"],
            pareto_pts["f1_event"],
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label="Pareto frontier",
        )

    plt.xlabel("Latency (ms) — mean time to first alert in GT incident")
    plt.ylabel("F1_event")
    plt.title("Temporal filter grid search (time_tune)")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PARETO_PNG_PATH, dpi=300)
    plt.close()

def main() -> None:
    run_grid_search()
    print(" Grid Search завершён. Результаты: test3_temporal_tuning/results/")

if __name__ == "__main__":
    main()
