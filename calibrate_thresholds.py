"""Calibration: datasets, feature extraction, EDA, threshold grid search."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import extract_calib_features
import prepare_datasets

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.py"

def class_stats(df: pd.DataFrame, feature: str) -> dict[str, float]:
    values = df[feature].dropna()
    if values.empty:
        return {key: float("nan") for key in ["mean", "median", "std", "q1", "q3", "iqr", "min", "max", "cv"]}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return {
        "mean": mean,
        "median": float(values.median()),
        "std": std,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(values.min()),
        "max": float(values.max()),
        "cv": float(std / mean) if mean != 0 else float("nan"),
    }

def kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(grid)
    bandwidth = max(np.std(values) * 0.2, 0.05)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(values.reshape(-1, 1))
    log_density = kde.score_samples(grid.reshape(-1, 1))
    density = np.exp(log_density)
    area = np.trapz(density, grid)
    return density / area if area > 0 else density

def distribution_metrics(abnormal_values: np.ndarray, normal_values: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    all_values = np.concatenate([abnormal_values, normal_values])
    if all_values.size < 2:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    grid = np.linspace(np.min(all_values), np.max(all_values), 512)
    d_abn = kde_density(abnormal_values, grid)
    d_norm = kde_density(normal_values, grid)

    overlap_coeff = float(np.trapz(np.minimum(d_abn, d_norm), grid))
    diff_idx = int(np.argmin(np.abs(d_abn - d_norm)))
    intersection = float(grid[diff_idx])
    if overlap_coeff <= 1e-6:
        intersection = float("nan")

    overlap_curve = np.minimum(d_abn, d_norm)
    mask = overlap_curve > (0.15 * np.max(overlap_curve))
    if np.any(mask):
        overlap_zone = (float(grid[np.where(mask)[0][0]]), float(grid[np.where(mask)[0][-1]]))
    else:
        overlap_zone = (float("nan"), float("nan"))
    return intersection, overlap_coeff, overlap_zone

def theta_precision_recall_bounds(df: pd.DataFrame) -> tuple[float, float]:
    thresholds = np.sort(df["theta"].dropna().unique())
    if thresholds.size == 0:
        return float("nan"), float("nan")
    best_precision_thr = float("nan")
    best_recall_thr = float("nan")
    gt = (df["gt_class"] == "ABNORMAL").astype(int).to_numpy()
    values = df["theta"].to_numpy()
    for thr in thresholds:
        pred = (values >= thr).astype(int)
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if np.isnan(best_precision_thr) and precision >= 0.9:
            best_precision_thr = float(thr)
        if recall >= 0.9:
            best_recall_thr = float(thr)
    return best_precision_thr, best_recall_thr

def safe_range(low: float, high: float, fallback_low: float, fallback_high: float) -> tuple[float, float]:
    if np.isnan(low):
        low = fallback_low
    if np.isnan(high):
        high = fallback_high
    if low > high:
        return min(low, high), max(low, high)
    return low, high

def render_feature_plot(
    df: pd.DataFrame,
    feature: str,
    output_path: Path,
    stats_abnormal: dict[str, float],
    stats_normal: dict[str, float],
    overlap_zone: tuple[float, float],
) -> None:
    abnormal = df[df["gt_class"] == "ABNORMAL"][feature].dropna().to_numpy()
    normal = df[df["gt_class"] == "NORMAL"][feature].dropna().to_numpy()
    if abnormal.size == 0 or normal.size == 0:
        return
    grid = np.linspace(min(np.min(abnormal), np.min(normal)), max(np.max(abnormal), np.max(normal)), 512)
    d_abn = kde_density(abnormal, grid)
    d_norm = kde_density(normal, grid)

    plt.figure(figsize=(10, 6))
    plt.hist(abnormal, bins=30, density=True, alpha=0.35, color="red", label="ABNORMAL")
    plt.hist(normal, bins=30, density=True, alpha=0.35, color="blue", label="NORMAL")
    plt.plot(grid, d_abn, color="darkred")
    plt.plot(grid, d_norm, color="navy")

    for stats, color in [(stats_abnormal, "darkred"), (stats_normal, "navy")]:
        plt.axvline(stats["median"], color=color, linestyle="--", linewidth=1.2)
        plt.axvline(stats["q1"], color=color, linestyle=":", linewidth=1.0)
        plt.axvline(stats["q3"], color=color, linestyle=":", linewidth=1.0)

    if not np.isnan(overlap_zone[0]) and not np.isnan(overlap_zone[1]):
        plt.axvspan(overlap_zone[0], overlap_zone[1], color="gray", alpha=0.15)

    plt.title(f"EDA {feature} distribution")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.grid(alpha=0.2)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def update_config_ranges(ranges: dict[str, float]) -> None:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    updates = {
        "GRID_THETA_MIN": int(round(ranges["GRID_THETA_MIN"])),
        "GRID_THETA_MAX": int(round(ranges["GRID_THETA_MAX"])),
        "GRID_THETA_STEP": 1,
        "GRID_K_BASE_MIN": round(ranges["GRID_K_BASE_MIN"], 3),
        "GRID_K_BASE_MAX": round(ranges["GRID_K_BASE_MAX"], 3),
        "GRID_K_BASE_STEP": 0.05,
        "GRID_K_CRITICAL_MIN": round(ranges["GRID_K_CRITICAL_MIN"], 3),
        "GRID_K_CRITICAL_MAX": round(ranges["GRID_K_CRITICAL_MAX"], 3),
        "GRID_K_CRITICAL_STEP": 0.05,
    }

    section_header = "# EDA-based ranges for Grid Search (exp A)"
    updated = original
    if section_header not in updated:
        updated += f"\n\n{section_header}\n"

    for key, value in updates.items():
        pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
        replacement = f"{key} = {repr(value)}"
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated)
        else:
            updated += f"{replacement}\n"

    CONFIG_PATH.write_text(updated, encoding="utf-8")
    print(
        "config.py обновлён: "
        f"GRID_THETA_MIN={updates['GRID_THETA_MIN']}, "
        f"GRID_THETA_MAX={updates['GRID_THETA_MAX']}, "
        f"GRID_K_BASE_MIN={updates['GRID_K_BASE_MIN']}, "
        f"GRID_K_BASE_MAX={updates['GRID_K_BASE_MAX']}, "
        f"GRID_K_CRITICAL_MAX={updates['GRID_K_CRITICAL_MAX']}"
    )

def exploratory_analysis(df: pd.DataFrame, experiment_name: str) -> dict[str, float]:
    abnormal_df = df[df["gt_class"] == "ABNORMAL"]
    normal_df = df[df["gt_class"] == "NORMAL"]

    theta_abn_stats = class_stats(abnormal_df, "theta")
    theta_norm_stats = class_stats(normal_df, "theta")
    k_abn_stats = class_stats(abnormal_df, "K_ar")
    k_norm_stats = class_stats(normal_df, "K_ar")

    theta_intersection, theta_overlap, theta_zone = distribution_metrics(
        abnormal_df["theta"].dropna().to_numpy(),
        normal_df["theta"].dropna().to_numpy(),
    )
    k_intersection, k_overlap, k_zone = distribution_metrics(
        abnormal_df["K_ar"].dropna().to_numpy(),
        normal_df["K_ar"].dropna().to_numpy(),
    )

    render_feature_plot(
        df,
        "theta",
        FIGURES_DIR / f"eda_theta_{experiment_name}.png",
        theta_abn_stats,
        theta_norm_stats,
        theta_zone,
    )
    render_feature_plot(
        df,
        "K_ar",
        FIGURES_DIR / f"eda_K_ar_{experiment_name}.png",
        k_abn_stats,
        k_norm_stats,
        k_zone,
    )

    theta_p90_thr, theta_r90_thr = theta_precision_recall_bounds(df)
    theta_min = max(theta_abn_stats["q1"], theta_p90_thr if not np.isnan(theta_p90_thr) else theta_abn_stats["q1"])
    theta_max = min(theta_norm_stats["q3"], theta_r90_thr if not np.isnan(theta_r90_thr) else theta_norm_stats["q3"])
    theta_min, theta_max = safe_range(theta_min, theta_max, theta_abn_stats["q1"], theta_norm_stats["q3"])

    k_base_min = k_abn_stats["median"] - 0.5 * k_abn_stats["iqr"]
    k_base_max = k_norm_stats["median"] + 0.5 * k_norm_stats["iqr"]
    k_base_min, k_base_max = safe_range(k_base_min, k_base_max, k_abn_stats["q1"], k_norm_stats["q3"])

    k_critical_min = k_base_min + 0.2
    k_critical_max = 1.2 * max(k_abn_stats["max"], k_norm_stats["max"])
    if k_critical_max <= k_critical_min:
        k_critical_max = k_critical_min + 0.2

    report = f"""# EDA Report ({experiment_name})

## Class Statistics

### ABNORMAL
- theta: mean={theta_abn_stats['mean']:.3f}, median={theta_abn_stats['median']:.3f}, std={theta_abn_stats['std']:.3f}, q1={theta_abn_stats['q1']:.3f}, q3={theta_abn_stats['q3']:.3f}, iqr={theta_abn_stats['iqr']:.3f}, min={theta_abn_stats['min']:.3f}, max={theta_abn_stats['max']:.3f}, cv={theta_abn_stats['cv']:.3f}
- K_ar: mean={k_abn_stats['mean']:.3f}, median={k_abn_stats['median']:.3f}, std={k_abn_stats['std']:.3f}, q1={k_abn_stats['q1']:.3f}, q3={k_abn_stats['q3']:.3f}, iqr={k_abn_stats['iqr']:.3f}, min={k_abn_stats['min']:.3f}, max={k_abn_stats['max']:.3f}, cv={k_abn_stats['cv']:.3f}

### NORMAL
- theta: mean={theta_norm_stats['mean']:.3f}, median={theta_norm_stats['median']:.3f}, std={theta_norm_stats['std']:.3f}, q1={theta_norm_stats['q1']:.3f}, q3={theta_norm_stats['q3']:.3f}, iqr={theta_norm_stats['iqr']:.3f}, min={theta_norm_stats['min']:.3f}, max={theta_norm_stats['max']:.3f}, cv={theta_norm_stats['cv']:.3f}
- K_ar: mean={k_norm_stats['mean']:.3f}, median={k_norm_stats['median']:.3f}, std={k_norm_stats['std']:.3f}, q1={k_norm_stats['q1']:.3f}, q3={k_norm_stats['q3']:.3f}, iqr={k_norm_stats['iqr']:.3f}, min={k_norm_stats['min']:.3f}, max={k_norm_stats['max']:.3f}, cv={k_norm_stats['cv']:.3f}

## Separation Analysis
- theta intersection: {theta_intersection:.3f}
- theta overlap coefficient: {theta_overlap:.3f}
- theta uncertainty zone: [{theta_zone[0]:.3f}, {theta_zone[1]:.3f}]
- K_ar intersection: {k_intersection:.3f}
- K_ar overlap coefficient: {k_overlap:.3f}
- K_ar uncertainty zone: [{k_zone[0]:.3f}, {k_zone[1]:.3f}]

## РЕКОМЕНДУЕМЫЕ ДИАПАЗОНЫ ДЛЯ GRID SEARCH ({experiment_name})
- theta_thr: [{theta_min:.0f}°, {theta_max:.0f}°] (точка пересечения: {theta_intersection:.0f}°)
- K_base: [{k_base_min:.3f}, {k_base_max:.3f}]
- K_critical: [{k_critical_min:.3f}, {k_critical_max:.3f}]

## ОБОСНОВАНИЕ
- Диапазон theta построен на Q1 ABNORMAL, Q3 NORMAL и порогах Precision/Recall >= 0.9.
- Диапазон K_base построен из медиан и IQR двух классов.
- Диапазон K_critical включает логический зазор +0.2 от K_base и верхний запас 1.2x от максимума.
- При отсутствии устойчивой точки пересечения используется fallback на квартильные границы.
"""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"eda_report_{experiment_name}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"{report_path} сохранён")

    return {
        "GRID_THETA_MIN": theta_min,
        "GRID_THETA_MAX": theta_max,
        "GRID_K_BASE_MIN": k_base_min,
        "GRID_K_BASE_MAX": k_base_max,
        "GRID_K_CRITICAL_MIN": k_critical_min,
        "GRID_K_CRITICAL_MAX": k_critical_max,
    }

def main() -> None:
    prepare_datasets.main()
    extract_calib_features.main()

    df_a = pd.read_csv(DATA_DIR / "features_predicted_A.csv")
    df_b = pd.read_csv(DATA_DIR / "features_predicted_B.csv")

    ranges_a = exploratory_analysis(df_a, experiment_name="A")
    exploratory_analysis(df_b, experiment_name="B")
    update_config_ranges(ranges_a)

    print("GRID_THETA_MIN =", int(round(ranges_a["GRID_THETA_MIN"])))
    print("GRID_THETA_MAX =", int(round(ranges_a["GRID_THETA_MAX"])))
    print("GRID_THETA_STEP = 1")
    print("GRID_K_BASE_MIN =", round(ranges_a["GRID_K_BASE_MIN"], 3))
    print("GRID_K_BASE_MAX =", round(ranges_a["GRID_K_BASE_MAX"], 3))
    print("GRID_K_CRITICAL_MAX =", round(ranges_a["GRID_K_CRITICAL_MAX"], 3))
    print("\nРАЗВЕДОЧНЫЙ АНАЛИЗ ЗАВЕРШЁН")

if __name__ == "__main__":
    main()
