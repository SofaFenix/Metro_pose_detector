"""Ablation summary for temporal filter configs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T2F_ROOT = PROJECT_ROOT / "test2_with_filters"
TABLE_CSV = T2F_ROOT / "results" / "comparison_table.csv"
ABLATION_MD = T2F_ROOT / "results" / "ablation_summary.md"

LABEL_A = "No filter (W=1)"
LABEL_C = "Optimized (Grid Search)"

def main() -> None:
    if not TABLE_CSV.is_file():
        raise FileNotFoundError(f"Отсутствует {TABLE_CSV}. Сначала calculate_metrics_comparison.py")

    df = pd.read_csv(TABLE_CSV)
    row_a = df.loc[df["configuration"] == LABEL_A].iloc[0]
    row_c = df.loc[df["configuration"] == LABEL_C].iloc[0]

    f1_a, f1_c = float(row_a["f1_event"]), float(row_c["f1_event"])
    lat_a = float(row_a["latency_ms"])
    lat_c = float(row_c["latency_ms"])
    far_a = float(row_a["far_alerts_per_min"])
    far_c = float(row_c["far_alerts_per_min"])

    delta_f1 = f1_c - f1_a
    delta_f1_pct = (delta_f1 / f1_a * 100.0) if f1_a > 0 else float("nan")
    delta_latency = lat_c - lat_a if np.isfinite(lat_a) and np.isfinite(lat_c) else float("nan")

    if far_a > 0 and far_c >= 0:
        far_reduction_factor = far_a / far_c if far_c > 0 else float("inf")
    elif far_a == 0 and far_c == 0:
        far_reduction_factor = 1.0
    else:
        far_reduction_factor = float("nan")

    conclusion = (
        f"Добавление временного фильтра (Optimized vs No filter) изменяет F1_event на "
        f"{delta_f1:+.4f} ({delta_f1_pct:+.1f}% относительно Config A)"
    )
    if np.isfinite(delta_latency):
        conclusion += f" при изменении средней задержки на {delta_latency:+.1f} мс."
    else:
        conclusion += "."

    lines = [
        "# Ablation summary (temporal filter)",
        "",
        "## Сравниваемые конфигурации",
        f"- **A:** {LABEL_A}",
        f"- **C:** {LABEL_C}",
        "",
        "## Прирост метрик (C − A)",
        "",
        f"| Метрика | Config A | Config C | Δ |",
        f"|---------|----------|----------|---|",
        f"| F1_event | {f1_a:.4f} | {f1_c:.4f} | {delta_f1:+.4f} |",
        f"| Latency (мс) | {lat_a:.1f} | {lat_c:.1f} | {delta_latency:+.1f} |" if np.isfinite(delta_latency) else "",
        f"| FAR (alerts/min) | {far_a:.4f} | {far_c:.4f} | {far_c - far_a:+.4f} |",
        "",
        "## Снижение FAR",
        f"- Отношение FAR(A) / FAR(C): **{far_reduction_factor:.2f}×** (если FAR(C)=0, см. таблицу)",
        "",
        "## Вывод",
        "",
        conclusion,
        "",
        "Финальный выбор параметров выполняется вручную по `comparison_table.md` и Pareto test3.",
        "",
    ]
    ABLATION_MD.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {ABLATION_MD}")

if __name__ == "__main__":
    main()
