# Selected temporal-filter parameters (Grid Search)

## Parameters

Pareto on (latency_ms, f1_event); among points with F1_event >= 95% of frontier max: min latency; tie-break: min window_ms.

- **WINDOW_MS** = `600`
- **MIN_ALERT_RATIO** = `0.70`
- Event recall / precision / F1: 0.5000 / 0.6667 / 0.5714
- Latency (mean): 1816.50 ms
- FAR (alerts/min on normal clips): 0.0000

## Top-3 combinations

1. W=600 ms, ratio=0.70 — F1=0.5714, Latency=1816.50 ms, Recall=0.5000, Prec=0.6667
2. W=700 ms, ratio=0.70 — F1=0.5714, Latency=1816.50 ms, Recall=0.5000, Prec=0.6667
3. W=700 ms, ratio=0.80 — F1=0.5714, Latency=1816.50 ms, Recall=0.5000, Prec=0.6667

## config.py
TEMPORAL_WINDOW_MS=600, MIN_ALERT_RATIO=0.7
