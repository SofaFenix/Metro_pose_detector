# EDA Report (B)

## Class Statistics

### ABNORMAL
- theta: mean=56.899, median=59.901, std=28.182, q1=39.919, q3=73.833, iqr=33.914, min=1.385, max=120.894, cv=0.495
- K_ar: mean=1.487, median=1.435, std=0.502, q1=1.130, q3=1.796, iqr=0.666, min=0.585, max=3.143, cv=0.338

### NORMAL
- theta: mean=39.508, median=32.023, std=29.376, q1=16.897, q3=63.367, iqr=46.470, min=1.621, max=103.215, cv=0.744
- K_ar: mean=0.509, median=0.502, std=0.128, q1=0.401, q3=0.623, iqr=0.222, min=0.277, max=0.747, cv=0.250

## Separation Analysis
- theta intersection: 45.821
- theta overlap coefficient: 0.635
- theta uncertainty zone: [1.385, 108.265]
- K_ar intersection: 0.793
- K_ar overlap coefficient: 0.072
- K_ar uncertainty zone: [0.473, 0.843]

## РЕКОМЕНДУЕМЫЕ ДИАПАЗОНЫ ДЛЯ GRID SEARCH (B)
- theta_thr: [9°, 104°] (точка пересечения: 46°)
- K_base: [0.613, 1.102]
- K_critical: [0.813, 3.772]

## ОБОСНОВАНИЕ
- Диапазон theta построен на Q1 ABNORMAL, Q3 NORMAL и порогах Precision/Recall >= 0.9.
- Диапазон K_base построен из медиан и IQR двух классов.
- Диапазон K_critical включает логический зазор +0.2 от K_base и верхний запас 1.2x от максимума.
- При отсутствии устойчивой точки пересечения используется fallback на квартильные границы.
