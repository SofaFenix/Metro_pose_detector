# Grid Search summary (time_tune)

Источник: `grid_search_time_tune.csv` (42 комбинаций)
График Pareto: `figures/pareto_frontier_time_tune.png`

## Топ-10 по F1_event

1. W=800 ms, ratio=0.7 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
2. W=600 ms, ratio=0.7 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
3. W=700 ms, ratio=0.7 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
4. W=700 ms, ratio=0.8 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
5. W=800 ms, ratio=0.8 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
6. W=400 ms, ratio=0.9 | F1=0.5455, Recall=0.5000, Prec=0.6000, Latency=1816.5 ms, FAR=0.000 alerts/min
7. W=500 ms, ratio=0.8 | F1=0.5455, Recall=0.5000, Prec=0.6000, Latency=1816.5 ms, FAR=0.000 alerts/min
8. W=500 ms, ratio=0.7 | F1=0.5455, Recall=0.5000, Prec=0.6000, Latency=1816.5 ms, FAR=0.000 alerts/min
9. W=300 ms, ratio=0.9 | F1=0.5455, Recall=0.5000, Prec=0.6000, Latency=1816.5 ms, FAR=0.000 alerts/min
10. W=600 ms, ratio=0.8 | F1=0.5455, Recall=0.5000, Prec=0.6000, Latency=1816.5 ms, FAR=0.000 alerts/min

## Топ-10 по минимальной Latency (мс)

1. W=300 ms, ratio=0.3 | F1=0.4615, Recall=0.5000, Prec=0.4286, Latency=1816.5 ms, FAR=0.000 alerts/min
2. W=300 ms, ratio=0.4 | F1=0.4615, Recall=0.5000, Prec=0.4286, Latency=1816.5 ms, FAR=0.000 alerts/min
3. W=300 ms, ratio=0.5 | F1=0.4615, Recall=0.5000, Prec=0.4286, Latency=1816.5 ms, FAR=0.000 alerts/min
4. W=300 ms, ratio=0.6 | F1=0.5000, Recall=0.5000, Prec=0.5000, Latency=1816.5 ms, FAR=0.000 alerts/min
5. W=300 ms, ratio=0.7 | F1=0.5000, Recall=0.5000, Prec=0.5000, Latency=1816.5 ms, FAR=0.000 alerts/min
6. W=300 ms, ratio=0.8 | F1=0.5000, Recall=0.5000, Prec=0.5000, Latency=1816.5 ms, FAR=0.000 alerts/min
7. W=300 ms, ratio=0.9 | F1=0.5455, Recall=0.5000, Prec=0.6000, Latency=1816.5 ms, FAR=0.000 alerts/min
8. W=400 ms, ratio=0.3 | F1=0.4615, Recall=0.5000, Prec=0.4286, Latency=1816.5 ms, FAR=0.000 alerts/min
9. W=400 ms, ratio=0.4 | F1=0.4615, Recall=0.5000, Prec=0.4286, Latency=1816.5 ms, FAR=0.000 alerts/min
10. W=400 ms, ratio=0.5 | F1=0.4615, Recall=0.5000, Prec=0.4286, Latency=1816.5 ms, FAR=0.000 alerts/min

## Граница Парето (недоминируемые точки)

1. W=600 ms, ratio=0.7 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
2. W=700 ms, ratio=0.7 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
3. W=700 ms, ratio=0.8 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
4. W=800 ms, ratio=0.7 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min
5. W=800 ms, ratio=0.8 | F1=0.5714, Recall=0.5000, Prec=0.6667, Latency=1816.5 ms, FAR=0.000 alerts/min

## Рекомендации

- **Баланс качества и скорости:** выберите точку на границе Пareto с приемлемой `latency_ms` и максимальным `f1_event` в её окрестности.
- **Минимальная задержка:** ориентируйтесь на топ по `latency_ms`, проверьте `event_recall` и `far_alerts_per_min` на нормальных роликах.
- **Максимальное покрытие инцидентов:** ориентируйтесь на топ по `f1_event` / `event_recall`, сверьте рост `far_alerts_per_min`.
- Финальные `WINDOW_MS` и `MIN_ALERT_RATIO` **не выбираются автоматически** — зафиксируйте вручную после просмотра CSV и графика Pareto.
