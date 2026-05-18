# Ablation summary (temporal filter)

## Сравниваемые конфигурации
- **A:** No filter (W=1)
- **C:** Optimized (Grid Search)

## Прирост метрик (C − A)

| Метрика | Config A | Config C | Δ |
|---------|----------|----------|---|
| F1_event | 0.8000 | 0.8889 | +0.0889 |
| Latency (мс) | 0.0 | 290.7 | +290.7 |
| FAR (alerts/min) | 264.1203 | 195.0427 | -69.0776 |

## Снижение FAR
- Отношение FAR(A) / FAR(C): **1.35×** (если FAR(C)=0, см. таблицу)

## Result

Добавление временного фильтра (Optimized vs No filter) изменяет F1_event на +0.0889 (+11.1% относительно Config A) при изменении средней задержки на +290.7 мс.

Финальный выбор параметров выполняется вручную по `comparison_table.md` и Pareto test3.
