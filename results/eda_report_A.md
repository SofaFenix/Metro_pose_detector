# EDA Report (A)

## Class Statistics

### ABNORMAL
- theta: mean=56.899, median=59.901, std=28.182, q1=39.919, q3=73.833, iqr=33.914, min=1.385, max=120.894, cv=0.495
- K_ar: mean=1.487, median=1.435, std=0.502, q1=1.130, q3=1.796, iqr=0.666, min=0.585, max=3.143, cv=0.338

### NORMAL
- theta: mean=42.761, median=34.741, std=32.238, q1=18.254, q3=63.594, iqr=45.339, min=1.769, max=117.909, cv=0.754
- K_ar: mean=0.587, median=0.591, std=0.165, q1=0.457, q3=0.728, iqr=0.272, min=0.287, max=0.881, cv=0.282

## Separation Analysis
- theta intersection: 102.652
- theta overlap coefficient: 0.618
- theta uncertainty zone: [1.385, 120.894]
- K_ar intersection: 0.902
- K_ar overlap coefficient: 0.117
- K_ar uncertainty zone: [0.499, 0.958]

## РЕКОМЕНДУЕМЫЕ ДИАПАЗОНЫ ДЛЯ GRID SEARCH (A)
- theta_thr: [9°, 119°] (точка пересечения: 103°)
- K_base: [0.727, 1.102]
- K_critical: [0.927, 3.772]

## ОБОСНОВАНИЕ
- Диапазон theta построен на Q1 ABNORMAL, Q3 NORMAL и порогах Precision/Recall >= 0.9.
- Диапазон K_base построен из медиан и IQR двух классов.
- Диапазон K_critical включает логический зазор +0.2 от K_base и верхний запас 1.2x от максимума.
- При отсутствии устойчивой точки пересечения используется fallback на квартильные границы.
