# Metrics report (final_test)

## Visibility filter
- Фильтр: `visibility_rate >= config.VISIBILITY_THRESHOLD` (**0.7**).
- Исключено из количественного анализа: **76** из **697** кадров.

## Labels
- **y** ∈ {0, 1}: GT `NORMAL` → 0; любая метка с подстрокой `ABNORMAL` → 1 (включая `ABNORMAL_HORIZONTAL_POSTURE`).
- **ŷ**: `alert_triggered` ∈ {0, 1}.

**FAR** = FP / (FP + TN).

## Metrics

| Метрика | Значение |
|---:|---:|
| Precision | 0.7066 |
| Recall | 0.8939 |
| F1-score | 0.7893 |
| FAR | 0.1002 |

## Confusion matrix

Изображение: `final_test/results/confusion_matrix.png`

```
[[440, 49], [14, 118]]
```
