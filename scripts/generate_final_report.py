"""Assemble final temporal-filter report markdown."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
T3_ROOT = PROJECT_ROOT / "test3_temporal_tuning"
T2F_ROOT = PROJECT_ROOT / "test2_with_filters"

GRID_CSV = T3_ROOT / "results" / "grid_search_time_tune.csv"
PARETO_PNG = T3_ROOT / "figures" / "pareto_frontier_time_tune.png"
SELECTED_MD = T3_ROOT / "results" / "selected_params.md"
COMPARISON_MD = T2F_ROOT / "results" / "comparison_table.md"
ABLATION_MD = T2F_ROOT / "results" / "ablation_summary.md"
OUTPUT_REPORT = T3_ROOT / "results" / "final_report.md"

def parse_selected(md_text: str) -> tuple[str, str]:
    wm = re.search(r"\*\*WINDOW_MS\*\*\s*=\s*`(\d+)`", md_text)
    mr = re.search(r"\*\*MIN_ALERT_RATIO\*\*\s*=\s*`([\d.]+)`", md_text)
    return (
        wm.group(1) if wm else "—",
        mr.group(1) if mr else "—",
    )

def main() -> None:
    sections: list[str] = []

    sections.append("## 1. Цель эксперимента\n")
    sections.append(
        "Подбор параметров временного фильтра (окно в миллисекундах и доля «сырых» "
        "аномальных кадров в окне) для zero-shot пайплайна детекции горизонтальной позы "
        "на ограниченной валидационной выборке, с последующей проверкой на альтернативном "
        "наборе кадров и сравнением с конфигурациями без фильтра и с фильтром из `config.py`.\n"
    )

    sections.append("## 2. Дизайн (Grid Search + Ablation)\n")
    sections.append(
        "- **Grid Search** выполнялся на выборке `time_tune` (`test3_temporal_tuning`): "
        "перебор `WINDOW_MS` и `MIN_ALERT_RATIO` при фиксированном `FALLBACK_K` из `config.py`; "
        "метрики на уровне событий (инцидентов).\n"
        "- **Ablation / сравнение** на `test2_with_filters`: три конфигурации — без временного "
        "окна, текущее окно в кадрах из `config.py`, параметры, согласованные с лучшими точками Grid Search.\n"
    )

    sections.append("## 3. Результаты Grid Search (time_tune)\n")
    if PARETO_PNG.is_file():
        rel = PARETO_PNG.relative_to(T3_ROOT).as_posix()
        sections.append(
            f"График компромисса задержки и качества (ось X — средняя задержка алерта в мс, "
            f"ось Y — `F1_event`): `{rel}`.\n"
        )
        sections.append(
            "Граница Парето отмечает набор недоминируемых комбинаций: улучшение одной из метрик "
            "(например, `F1_event`) без ухудшения другой (`latency_ms`) невозможно в пределах сетки.\n"
        )
    else:
        sections.append("График: отсутствует `figures/pareto_frontier_time_tune.png`.\n")

    if GRID_CSV.is_file():
        g = pd.read_csv(GRID_CSV)
        top_f1 = g.nlargest(5, "f1_event")[
            ["window_ms", "min_alert_ratio", "f1_event", "latency_ms", "event_recall", "event_precision"]
        ]
        g_lat = g[np.isfinite(g["latency_ms"])].nsmallest(5, "latency_ms")[
            ["window_ms", "min_alert_ratio", "f1_event", "latency_ms", "event_recall", "event_precision"]
        ]
        sections.append("### Топ-5 по F1_event\n\n```\n")
        sections.append(top_f1.to_string(index=False))
        sections.append("\n```\n\n### Топ-5 по минимальной Latency (мс)\n\n```\n")
        sections.append(g_lat.to_string(index=False))
        sections.append("\n```\n")
    else:
        sections.append("Файл `grid_search_time_tune.csv`: отсутствует.\n")

    sections.append("## 4. Сравнение на валидационной выборке (test2)\n")
    if COMPARISON_MD.is_file():
        sections.append("Сводная таблица (`test2_with_filters/results/comparison_table.md`):\n\n")
        sections.append(COMPARISON_MD.read_text(encoding="utf-8"))
        sections.append("\n")
    else:
        sections.append("Файл `comparison_table.md`: отсутствует.\n")

    if ABLATION_MD.is_file():
        sections.append("### Прирост метрик при добавлении фильтра (фрагмент ablation)\n\n")
        sections.append(ABLATION_MD.read_text(encoding="utf-8"))
        sections.append("\n")

    sections.append("## 5. Рекомендованные параметры\n")
    if SELECTED_MD.is_file():
        sel_txt = SELECTED_MD.read_text(encoding="utf-8")
        wm, mr = parse_selected(sel_txt)
        sections.append(
            f"На основе границы Парето и правила «максимальный `F1_event` среди точек с "
            f"`F1` не ниже 95% от максимума на фронте, затем минимальная задержка» рекомендуется:\n\n"
            f"- `WINDOW_MS` = **{wm}**\n"
            f"- `MIN_ALERT_RATIO` = **{mr}**\n\n"
            "Обоснование: выбор ограничен недоминируемыми точками сетки; дополнительный порог по "
            "`F1_event` задаёт допустимое отклонение от лучшего качества на фронте в пользу меньшей задержки.\n\n"
            "Полная запись: `test3_temporal_tuning/results/selected_params.md`.\n"
        )
    else:
        sections.append("Файл `selected_params.md`: отсутствует.\n")

    sections.append(
        "### Параметры config.py\n\n"
        "`TEMPORAL_WINDOW_MS`, `MIN_ALERT_RATIO` — значения из `selected_params.md`.\n"
    )

    sections.append("## 6. Ограничения\n")
    sections.append(
        "- Результаты получены на ограниченной выборке (набор `time_tune`, восемь роликов, "
        "и воспроизведение на кадрах `test2_with_filters`).\n"
        "- Для промышленного внедрения требуется валидация на независимом тестовом наборе "
        "и учёт сдвига распределения (domain gap).\n"
        "- Метрики событийной выборки чувствительны к определению границ инцидента в разметке "
        "и к порогу видимости ключевых точек при окклюзии.\n"
    )

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("# Final report — temporal filter study\n\n" + "".join(sections), encoding="utf-8")
    print(f"[OK] {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
