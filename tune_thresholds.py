"""Grid search THETA, K_BASE, K_CRITICAL; writes calibration_report.md and config.py."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIG_PATH = PROJECT_ROOT / "config.py"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "experiment_log.md"

def parse_eda_fallback(eda_report_path: Path) -> dict[str, float]:
    text = eda_report_path.read_text(encoding="utf-8")
    theta_match = re.search(r"theta_thr:\s*\[(\d+(?:\.\d+)?)°,\s*(\d+(?:\.\d+)?)°\]", text)
    k_base_match = re.search(r"K_base:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]", text)
    k_critical_match = re.search(r"K_critical:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]", text)
    if not (theta_match and k_base_match and k_critical_match):
        raise ValueError(f"EDA fallback parse failed: {eda_report_path}")
    return {
        "GRID_THETA_MIN": float(theta_match.group(1)),
        "GRID_THETA_MAX": float(theta_match.group(2)),
        "GRID_THETA_STEP": 1.0,
        "GRID_K_BASE_MIN": float(k_base_match.group(1)),
        "GRID_K_BASE_MAX": float(k_base_match.group(2)),
        "GRID_K_BASE_STEP": 0.05,
        "GRID_K_CRITICAL_MIN": float(k_critical_match.group(1)),
        "GRID_K_CRITICAL_MAX": float(k_critical_match.group(2)),
        "GRID_K_CRITICAL_STEP": 0.05,
    }

def grid_params_with_fallback() -> dict[str, float]:
    keys = [
        "GRID_THETA_MIN",
        "GRID_THETA_MAX",
        "GRID_THETA_STEP",
        "GRID_K_BASE_MIN",
        "GRID_K_BASE_MAX",
        "GRID_K_BASE_STEP",
        "GRID_K_CRITICAL_MIN",
        "GRID_K_CRITICAL_MAX",
        "GRID_K_CRITICAL_STEP",
    ]
    has_all = all(hasattr(config, key) for key in keys)
    if has_all:
        return {key: float(getattr(config, key)) for key in keys}
    fallback = parse_eda_fallback(RESULTS_DIR / "eda_report_A.md")
    fallback_b = parse_eda_fallback(RESULTS_DIR / "eda_report_B.md")
    fallback["GRID_THETA_MAX"] = max(fallback["GRID_THETA_MAX"], fallback_b["GRID_THETA_MAX"])
    fallback["GRID_K_BASE_MIN"] = min(fallback["GRID_K_BASE_MIN"], fallback_b["GRID_K_BASE_MIN"])
    fallback["GRID_K_BASE_MAX"] = max(fallback["GRID_K_BASE_MAX"], fallback_b["GRID_K_BASE_MAX"])
    fallback["GRID_K_CRITICAL_MAX"] = max(fallback["GRID_K_CRITICAL_MAX"], fallback_b["GRID_K_CRITICAL_MAX"])
    return fallback

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp_rate,
    }

def auto_tune_thresholds(df: pd.DataFrame, experiment_name: str, params: dict[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{experiment_name}: empty features dataframe")
    required_cols = {"gt_class", "theta", "K_ar"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{experiment_name}: missing columns {missing}")

    y_true = (df["gt_class"].astype(str) == "ABNORMAL").astype(int).to_numpy()
    theta_values = df["theta"].to_numpy(dtype=float)
    k_ar_values = df["K_ar"].to_numpy(dtype=float)
    valid = np.isfinite(theta_values) & np.isfinite(k_ar_values)
    y_true = y_true[valid]
    theta_values = theta_values[valid]
    k_ar_values = k_ar_values[valid]
    if y_true.size == 0:
        raise ValueError(f"{experiment_name}: all rows invalid after NaN filtering")

    theta_range = np.arange(params["GRID_THETA_MIN"], params["GRID_THETA_MAX"] + params["GRID_THETA_STEP"], params["GRID_THETA_STEP"])
    k_base_range = np.arange(params["GRID_K_BASE_MIN"], params["GRID_K_BASE_MAX"] + params["GRID_K_BASE_STEP"], params["GRID_K_BASE_STEP"])
    k_critical_max = params["GRID_K_CRITICAL_MAX"]
    k_critical_step = params["GRID_K_CRITICAL_STEP"]

    f1_heatmap = np.full((len(theta_range), len(k_base_range)), np.nan, dtype=float)
    best: dict[str, Any] | None = None

    for i, theta_thr in enumerate(theta_range):
        for j, k_base in enumerate(k_base_range):
            k_critical_min = max(k_base + 0.2, params["GRID_K_CRITICAL_MIN"])
            k_critical_range = np.arange(k_critical_min, k_critical_max + k_critical_step, k_critical_step)
            if k_critical_range.size == 0:
                continue
            best_f1_for_cell = -1.0

            for k_critical in k_critical_range:
                y_pred = ((theta_values > theta_thr) & (k_ar_values > k_base)) | (k_ar_values > k_critical)
                y_pred_int = y_pred.astype(int)
                metrics = calc_metrics(y_true, y_pred_int)
                candidate = {
                    "experiment": experiment_name,
                    "theta_threshold": float(theta_thr),
                    "k_base": float(k_base),
                    "k_critical": float(k_critical),
                    **metrics,
                }

                if metrics["f1"] > best_f1_for_cell:
                    best_f1_for_cell = metrics["f1"]

                if best is None:
                    best = candidate
                else:
                    better_f1 = candidate["f1"] > best["f1"]
                    tie_break = np.isclose(candidate["f1"], best["f1"]) and candidate["fp"] < best["fp"]
                    if better_f1 or tie_break:
                        best = candidate

            f1_heatmap[i, j] = best_f1_for_cell if best_f1_for_cell >= 0 else np.nan

    if best is None:
        raise RuntimeError(f"{experiment_name}: no valid parameter combination found")
    return best, f1_heatmap

def save_grid_results(path: Path, best: dict[str, Any], params: dict[str, float]) -> None:
    payload = {"best": best, "grid_params": params}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def save_heatmap(heatmap: np.ndarray, params: dict[str, float], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Best F1")
    theta_ticks = np.linspace(0, heatmap.shape[0] - 1, min(10, heatmap.shape[0]), dtype=int)
    k_ticks = np.linspace(0, heatmap.shape[1] - 1, min(10, heatmap.shape[1]), dtype=int)
    theta_vals = np.arange(params["GRID_THETA_MIN"], params["GRID_THETA_MAX"] + params["GRID_THETA_STEP"], params["GRID_THETA_STEP"])
    k_vals = np.arange(params["GRID_K_BASE_MIN"], params["GRID_K_BASE_MAX"] + params["GRID_K_BASE_STEP"], params["GRID_K_BASE_STEP"])
    plt.yticks(theta_ticks, [f"{theta_vals[idx]:.0f}" for idx in theta_ticks])
    plt.xticks(k_ticks, [f"{k_vals[idx]:.2f}" for idx in k_ticks], rotation=45)
    plt.xlabel("K_base")
    plt.ylabel("theta_threshold")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def update_config_thresholds(selected: dict[str, Any], source_experiment: str) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    replacements = {
        "THETA_THRESHOLD": f"# Auto-calibrated via Grid Search ({source_experiment})\nTHETA_THRESHOLD = {int(round(selected['theta_threshold']))}",
        "K_BASE": f"# Auto-calibrated via Grid Search ({source_experiment})\nK_BASE = {selected['k_base']:.3f}",
        "K_CRITICAL": f"# Auto-calibrated via Grid Search ({source_experiment})\nK_CRITICAL = {selected['k_critical']:.3f}",
    }

    for key, repl in replacements.items():
        pattern = re.compile(rf"(^\s*#\s*Auto-calibrated via Grid Search.*\n)?^\s*{key}\s*=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(repl, text)
        else:
            text += f"\n{repl}\n"
    CONFIG_PATH.write_text(text, encoding="utf-8")

def append_calibration_report(best_a: dict[str, Any], best_b: dict[str, Any], selected: dict[str, Any], source_experiment: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "calibration_report.md"
    report = f"""# Grid Search Calibration Report

## Summary Table

| Эксперимент | T_theta | K_base | K_critical | F1-score | FP rate |
|---|---:|---:|---:|---:|---:|
| A | {best_a['theta_threshold']:.0f} | {best_a['k_base']:.3f} | {best_a['k_critical']:.3f} | {best_a['f1']:.4f} | {best_a['fp_rate']:.4f} |
| B | {best_b['theta_threshold']:.0f} | {best_b['k_base']:.3f} | {best_b['k_critical']:.3f} | {best_b['f1']:.4f} | {best_b['fp_rate']:.4f} |

## Selected Thresholds
- Source experiment: {source_experiment}
- THETA_THRESHOLD = {selected['theta_threshold']:.0f}
- K_BASE = {selected['k_base']:.3f}
- K_CRITICAL = {selected['k_critical']:.3f}
- F1-score = {selected['f1']:.4f}
- FP rate = {selected['fp_rate']:.4f}
"""
    report_path.write_text(report, encoding="utf-8")

def choose_best(best_a: dict[str, Any], best_b: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if best_a["f1"] > best_b["f1"]:
        return best_a, "exp A"
    if best_b["f1"] > best_a["f1"]:
        return best_b, "exp B"
    if best_a["fp_rate"] <= best_b["fp_rate"]:
        return best_a, "exp A"
    return best_b, "exp B"

def append_experiment_log_entry(best_a: dict[str, Any], best_b: dict[str, Any], selected_source: str) -> None:
    if not EXPERIMENT_LOG_PATH.exists():
        return
    line = (
        f"| {date.today().isoformat()} | exp8_grid_search_run | Выполнен grid search для A/B, "
        f"обновлены THETA/K_BASE/K_CRITICAL в config.py | "
        f"A_F1={best_a['f1']:.4f}; B_F1={best_b['f1']:.4f} | completed | "
        f"Выбран {selected_source}; отчёт: `results/calibration_report.md` |\n"
    )
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)

def main() -> None:
    params = grid_params_with_fallback()
    path_a = DATA_DIR / "features_predicted_A.csv"
    path_b = DATA_DIR / "features_predicted_B.csv"
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError("Missing features CSV files. Run extract_calib_features.py first.")

    try:
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
    except Exception as exc:
        raise RuntimeError(f"Failed to read features CSV: {exc}") from exc

    best_a, heatmap_a = auto_tune_thresholds(df_a, "A", params)
    best_b, heatmap_b = auto_tune_thresholds(df_b, "B", params)

    save_grid_results(RESULTS_DIR / "grid_search_A.json", best_a, params)
    save_grid_results(RESULTS_DIR / "grid_search_B.json", best_b, params)
    save_heatmap(heatmap_a, params, FIGURES_DIR / "f1_heatmap_A.png", "F1 heatmap (A)")
    save_heatmap(heatmap_b, params, FIGURES_DIR / "f1_heatmap_B.png", "F1 heatmap (B)")

    selected, selected_source = choose_best(best_a, best_b)
    update_config_thresholds(selected, selected_source)
    append_calibration_report(best_a, best_b, selected, selected_source)
    append_experiment_log_entry(best_a, best_b, selected_source)

    print(
        f"Пороги обновлены: T_theta={selected['theta_threshold']:.0f}, "
        f"K_base={selected['k_base']:.3f}, K_critical={selected['k_critical']:.3f}"
    )
    print("\nЭксперимент | T_theta | K_base | K_critical | F1-score | FP rate")
    print(
        f"A | {best_a['theta_threshold']:.0f} | {best_a['k_base']:.3f} | {best_a['k_critical']:.3f} | "
        f"{best_a['f1']:.4f} | {best_a['fp_rate']:.4f}"
    )
    print(
        f"B | {best_b['theta_threshold']:.0f} | {best_b['k_base']:.3f} | {best_b['k_critical']:.3f} | "
        f"{best_b['f1']:.4f} | {best_b['fp_rate']:.4f}"
    )

if __name__ == "__main__":
    main()
