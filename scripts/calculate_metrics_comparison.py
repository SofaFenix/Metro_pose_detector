"""Event metrics comparison for filter configurations."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "test2_with_filters" / "scripts"
T2F_ROOT = PROJECT_ROOT / "test2_with_filters"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402

from comparison_core import GT_CSV_PATH, compute_event_metrics  # noqa: E402

RESULTS_DIR = T2F_ROOT / "results"
FIGURES_DIR = T2F_ROOT / "figures"

PREDICTION_FILES = {
    "No filter (W=1)": "predictions_A_no_filter.csv",
    "Current config": "predictions_B_current_config.csv",
    "Optimized (Grid Search)": "predictions_C_optimized.csv",
}

TABLE_CSV = RESULTS_DIR / "comparison_table.csv"
TABLE_MD = RESULTS_DIR / "comparison_table.md"
CHART_PNG = FIGURES_DIR / "metrics_comparison.png"

def main() -> None:
    gt_df = pd.read_csv(GT_CSV_PATH)
    rows: list[dict[str, object]] = []

    for label, fname in PREDICTION_FILES.items():
        path = RESULTS_DIR / fname
        if not path.is_file():
            raise FileNotFoundError(f"Отсутствует {path}. Сначала run_comparison.py")
        pred_df = pd.read_csv(path)
        m = compute_event_metrics(pred_df, gt_df)
        rows.append(
            {
                "configuration": label,
                "event_recall": m["event_recall"],
                "event_precision": m["event_precision"],
                "f1_event": m["f1_event"],
                "latency_ms": m["latency_ms"],
                "far_alerts_per_min": m["far_alerts_per_min"],
                "excluded_visibility_frames": m["excluded_visibility_frames"],
                "total_frames": m["total_frames"],
            }
        )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_CSV, index=False, encoding="utf-8")

    md_lines = [
        "# Comparison table (test2_with_filters)",
        "",
        f"Visibility threshold (reporting): `config.VISIBILITY_THRESHOLD` = {config.VISIBILITY_THRESHOLD}",
        "",
        "| Конфигурация | Event_Recall | Event_Precision | F1_event | Latency (мс) | FAR | Excluded (vis) |",
        "|--------------|--------------|-----------------|----------|--------------|-----|----------------|",
    ]
    for _, r in df.iterrows():
        lat = f"{r['latency_ms']:.1f}" if np.isfinite(r["latency_ms"]) else "n/a"
        md_lines.append(
            f"| {r['configuration']} | {r['event_recall']:.4f} | {r['event_precision']:.4f} | "
            f"{r['f1_event']:.4f} | {lat} | {r['far_alerts_per_min']:.4f} | "
            f"{int(r['excluded_visibility_frames'])} / {int(r['total_frames'])} |"
        )
    TABLE_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    _plot_metrics(df)
    print("Сравнение завершено. Таблица: test2_with_filters/results/comparison_table.md")

def _plot_metrics(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["event_recall", "event_precision", "f1_event", "far_alerts_per_min"]
    plot_df = df.melt(
        id_vars=["configuration"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="configuration")
    plt.ylabel("Value")
    plt.xlabel("Metric")
    plt.title("Temporal filter comparison (test2)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_PNG, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
