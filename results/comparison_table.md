# Comparison table (test2_with_filters)

Visibility threshold (reporting): `config.VISIBILITY_THRESHOLD` = 0.7

| Конфигурация | Event_Recall | Event_Precision | F1_event | Latency (мс) | FAR | Excluded (vis) |
|--------------|--------------|-----------------|----------|--------------|-----|----------------|
| No filter (W=1) | 1.0000 | 0.6667 | 0.8000 | 0.0 | 264.1203 | 76 / 697 |
| Current config | 1.0000 | 0.5714 | 0.7273 | 75.0 | 284.4372 | 76 / 697 |
| Optimized (Grid Search) | 1.0000 | 0.8000 | 0.8889 | 290.7 | 195.0427 | 76 / 697 |
