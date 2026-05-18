# Metrics report (test2)

## Visibility filter
- Фильтр: `visibility_rate >= config.VISIBILITY_THRESHOLD` (**0.7**).
- Исключено из количественного анализа: **76** из **697** кадров.

## Labels
- GT NORMAL→0, ABNORMAL→1; pred: alert_triggered.

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** = 2 · Precision · Recall / (Precision + Recall)
- **FAR** (False Alarm Rate) = FP / (FP + TN)

## Metrics

| Метрика | Значение |
|---:|---:|
| Precision | 0.7052 |
| Recall | 0.9242 |
| F1-score | 0.8000 |
| FAR | 0.1043 |

## Confusion matrix (строки: GT 0/1; столбцы: Pred 0/1)

```
[[438, 51], [10, 122]]
```

Изображение: `test2/results/confusion_matrix.png`
