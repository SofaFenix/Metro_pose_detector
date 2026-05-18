"""Summarize temporal grid search CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"

RESULTS_CSV_PATH = T3_ROOT / "results" / "grid_search_time_tune.csv"
SUMMARY_MD_PATH = T3_ROOT / "results" / "gridsearch_summary.md"
PARETO_PNG_PATH = T3_ROOT / "figures" / "pareto_frontier_time_tune.png"

def format_row(r: pd.Series) -> str:
    return (
        f"W={int(r['window_ms'])} ms, ratio={r['min_alert_ratio']:.1f} | "
        f"F1={r['f1_event']:.4f}, Recall={r['event_recall']:.4f}, "
        f"Prec={r['event_precision']:.4f}, Latency={r['latency_ms']:.1f} ms, "
        f"FAR={r['far_alerts_per_min']:.3f} alerts/min"
    )

def main() -> None:
    if not RESULTS_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Отсутствует {RESULTS_CSV_PATH}; зависимость: test3_temporal_tuning/scripts/tune_temporal_gridsearch.py"
        )

    df = pd.read_csv(RESULTS_CSV_PATH)
    df_valid_latency = df[df["latency_ms"].notna()].copy()

    top_f1 = df.sort_values("f1_event", ascending=False).head(10)
    top_latency = df_valid_latency.sort_values("latency_ms", ascending=True).head(10)
    pareto = df.loc[df["is_pareto"] == True].sort_values(  # noqa: E712
        ["f1_event", "latency_ms"], ascending=[False, True]
    )

    lines = [
        "# Grid Search summary (time_tune)",
        "",
        f"Источник: `{RESULTS_CSV_PATH.name}` ({len(df)} комбинаций)",
        f"График Pareto: `{PARETO_PNG_PATH.relative_to(T3_ROOT).as_posix()}`",
        "",
        "## Топ-10 по F1_event",
        "",
    ]
    for i, (_, r) in enumerate(top_f1.iterrows(), 1):
        lines.append(f"{i}. {format_row(r)}")

    lines.extend(["", "## Топ-10 по минимальной Latency (мс)", ""])
    for i, (_, r) in enumerate(top_latency.iterrows(), 1):
        lines.append(f"{i}. {format_row(r)}")

    lines.extend(["", "## Граница Парето (недоминируемые точки)", ""])
    if pareto.empty:
        lines.append("_Нет точек на границе Парето._")
    else:
        for i, (_, r) in enumerate(pareto.iterrows(), 1):
            lines.append(f"{i}. {format_row(r)}")

    lines.extend(
        [
            "",
            "## Рекомендации",
            "",
            "- **Баланс качества и скорости:** выберите точку на границе Пareto с приемлемой `latency_ms` "
            "и максимальным `f1_event` в её окрестности.",
            "- **Минимальная задержка:** ориентируйтесь на топ по `latency_ms`, проверьте `event_recall` "
            "и `far_alerts_per_min` на нормальных роликах.",
            "- **Максимальное покрытие инцидентов:** ориентируйтесь на топ по `f1_event` / `event_recall`, "
            "сверьте рост `far_alerts_per_min`.",
            "- Финальные `WINDOW_MS` и `MIN_ALERT_RATIO` **не выбираются автоматически** — "
            "зафиксируйте вручную после просмотра CSV и графика Pareto.",
            "",
        ]
    )

    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Сводка: {SUMMARY_MD_PATH}")

if __name__ == "__main__":
    main()
