"""Run three temporal-filter configurations on test2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "test2_with_filters" / "scripts"
T2F_ROOT = PROJECT_ROOT / "test2_with_filters"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from comparison_core import (  # noqa: E402
    GT_CSV_PATH,
    apply_config,
    run_raw_feature_extraction,
)

RESULTS_DIR = T2F_ROOT / "results"

CONFIGS = {
    "A": {
        "label": "No filter (frame-level)",
        "file": "predictions_A_no_filter.csv",
    },
    "B": {
        "label": f"Current config (W={None})",  # filled at runtime
        "file": "predictions_B_current_config.csv",
    },
    "C": {
        "label": "Optimized (Grid Search)",
        "file": "predictions_C_optimized.csv",
    },
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run comparison pipeline (3 temporal configs)")
    p.add_argument("--optimized-window-ms", type=float, default=800.0, help="Config C: WINDOW_MS")
    p.add_argument("--optimized-min-ratio", type=float, default=0.7, help="Config C: MIN_ALERT_RATIO")
    p.add_argument("--no-cache", action="store_true", help="Пересчитать raw_features.csv")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if not GT_CSV_PATH.is_file():
        raise FileNotFoundError(f"Отсутствует GT: {GT_CSV_PATH}. ")

    import config  # noqa: WPS433

    gt_df = pd.read_csv(GT_CSV_PATH)
    raw_df = run_raw_feature_extraction(gt_df, use_cache=not args.no_cache)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str, float, float]] = [
        ("A", CONFIGS["A"]["file"], 1.0, 0.0),
        ("B", CONFIGS["B"]["file"], float(config.WINDOW_SIZE), float(config.MIN_ALERT_FRAMES)),
        ("C", CONFIGS["C"]["file"], float(args.optimized_window_ms), float(args.optimized_min_ratio)),
    ]

    for key, out_name, w_param, ratio_param in specs:
        if key == "B":
            print(
                f"[Config B] WINDOW_SIZE={config.WINDOW_SIZE}, "
                f"MIN_ALERT_FRAMES={config.MIN_ALERT_FRAMES}, FALLBACK_K={config.FALLBACK_K}"
            )
        elif key == "A":
            print("[Config A] No temporal filter (frame-level + fallback)")
        else:
            print(f"[Config C] WINDOW_MS={w_param}, MIN_ALERT_RATIO={ratio_param}")

        pred_df = apply_config(
            raw_df,
            key,
            window_ms=w_param if key == "C" else 1.0,
            min_alert_ratio=ratio_param if key == "C" else 0.0,
        )
        out_path = RESULTS_DIR / out_name
        pred_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  -> saved {out_path.name} ({len(pred_df)} rows)")

    print("Сравнение пайплайна завершено. Далее: calculate_metrics_comparison.py")

if __name__ == "__main__":
    main()
