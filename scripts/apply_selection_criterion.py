"""Pareto selection for temporal filter parameters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"

GRID_CSV = T3_ROOT / "results" / "grid_search_time_tune.csv"
OUTPUT_MD = T3_ROOT / "results" / "selected_params.md"

def pareto_mask(df: pd.DataFrame) -> pd.Series:
    """Доминирование: выше F1_event и ниже latency_ms."""
    x = df["latency_ms"].to_numpy(dtype=float)
    y = df["f1_event"].to_numpy(dtype=float)
    n = len(df)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            mask[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            if (y[j] >= y[i] and x[j] <= x[i]) and (y[j] > y[i] or x[j] < x[i]):
                mask[i] = False
                break
    return pd.Series(mask, index=df.index)

def main() -> None:
    if not GRID_CSV.is_file():
        raise FileNotFoundError(f"Отсутствует {GRID_CSV}")

    df = pd.read_csv(GRID_CSV)
    if "is_pareto" in df.columns:
        pareto_df = df.loc[df["is_pareto"].astype(bool)].copy()
    else:
        pareto_df = df.loc[pareto_mask(df)].copy()

    if pareto_df.empty:
        pareto_df = df[np.isfinite(df["latency_ms"])].copy()

    max_f1 = float(pareto_df["f1_event"].max())
    eligible = pareto_df.loc[
        (pareto_df["f1_event"] >= max_f1 * 0.95) & np.isfinite(pareto_df["latency_ms"])
    ].copy()
    if eligible.empty:
        eligible = pareto_df.loc[np.isfinite(pareto_df["latency_ms"])].copy()

    ranked = eligible.sort_values(
        ["f1_event", "latency_ms", "window_ms"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    top3 = ranked.head(3)
    primary = ranked.iloc[0]

    lines_md = [
        "# Selected temporal-filter parameters (Grid Search)",
        "",
        "## Primary recommendation",
        "",
        "Критерий: граница Парето по (`latency_ms`, `f1_event`); среди точек с "
        "`F1_event` не ниже 95% от максимума на фронте — минимальная задержка; "
        "при равенстве — меньший `window_ms`.",
        "",
        f"- **WINDOW_MS** = `{int(primary['window_ms'])}`",
        f"- **MIN_ALERT_RATIO** = `{float(primary['min_alert_ratio']):.2f}`",
        f"- Event recall / precision / F1: {primary['event_recall']:.4f} / {primary['event_precision']:.4f} / {primary['f1_event']:.4f}",
        f"- Latency (mean): {primary['latency_ms']:.2f} ms",
        f"- FAR (alerts/min on normal clips): {primary['far_alerts_per_min']:.4f}",
        "",
        "## Top-3 combinations",
        "",
    ]
    for i, row in top3.iterrows():
        lines_md.append(
            f"{i + 1}. W={int(row['window_ms'])} ms, ratio={float(row['min_alert_ratio']):.2f} — "
            f"F1={row['f1_event']:.4f}, Latency={row['latency_ms']:.2f} ms, "
            f"Recall={row['event_recall']:.4f}, Prec={row['event_precision']:.4f}"
        )

    lines_md.extend(
        [
            "",
            "## Manual application",
            "",
            "```text",
            "# Пример плейсхолдеров для ручного внесения (адаптируйте к именам в вашем коде):",
            "# TEMPORAL_WINDOW_MS = <WINDOW_MS>",
            "# MIN_ALERT_RATIO = <MIN_ALERT_RATIO>",
            "```",
            "",
            "`config.py` данным скриптом не изменяется.",
            "",
        ]
    )

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines_md), encoding="utf-8")

    print("=== TOP-3 recommended (Grid Search, time_tune) ===")
    for i, row in top3.iterrows():
        print(
            f"{i + 1}. WINDOW_MS={int(row['window_ms'])}, MIN_ALERT_RATIO={float(row['min_alert_ratio']):.2f} | "
            f"F1={row['f1_event']:.4f}, Latency={row['latency_ms']:.1f} ms, "
            f"R={row['event_recall']:.4f}, P={row['event_precision']:.4f}, FAR={row['far_alerts_per_min']:.4f}"
        )

    print("\n=== Suggested manual placeholders ===")
    print(f"# TEMPORAL_WINDOW_MS = {int(primary['window_ms'])}")
    print(f"# MIN_ALERT_RATIO = {float(primary['min_alert_ratio']):.2f}")
    print("# # config.py: TEMPORAL_WINDOW_MS, MIN_ALERT_RATIO")
    print("\nconfig.py не был изменён автоматически. Примените параметры вручную, если согласны.")
    print(f"\nSaved: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
