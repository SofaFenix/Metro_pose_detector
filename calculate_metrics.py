"""Metrics and confusion matrix from predictions CSV."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

import config

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV_PATH = PROJECT_ROOT / "data" / "predictions.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"
METRICS_MD_PATH = RESULTS_DIR / "metrics_report.md"

def classify_gt_binary(series: pd.Series) -> np.ndarray:
    """GT: NORMAL -> 0, ABNORMAL_FLOOR_POSTURE -> 1."""
    gt = np.full(len(series), fill_value=-1, dtype=np.int64)
    gt[(series == "NORMAL").to_numpy()] = 0
    gt[(series == "ABNORMAL_FLOOR_POSTURE").to_numpy()] = 1
    return gt

def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FAR = FP / (FP + TN)."""
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def main() -> None:
    if not PREDICTIONS_CSV_PATH.exists():
        raise FileNotFoundError(f"Нет файла: {PREDICTIONS_CSV_PATH}. Сначала run_pipeline_inference.py")

    df = pd.read_csv(PREDICTIONS_CSV_PATH)
    total_rows = len(df)
    thresh = float(config.VISIBILITY_THRESHOLD)

    excluded_mask = df["visibility_rate"] < thresh
    excluded_count = int(excluded_mask.sum())
    df_vis = df.loc[~excluded_mask].reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if df_vis.empty:
        note = (
            f"# Metrics report\n\n**Нет строк после фильтра visibility >= {thresh}.**\n\n"
            f"| Исключено | {excluded_count} / {total_rows} |\n"
        )
        METRICS_MD_PATH.write_text(note, encoding="utf-8")
        print("Нет кадров с visibility_rate >= {:.2f}".format(thresh))
        print(f"excluded_count: {excluded_count} / {total_rows}")
        return

    y_true = classify_gt_binary(df_vis["gt_class"])
    if np.any(y_true < 0):
        raise ValueError("В GT встречаются классы помимо NORMAL / ABNORMAL_FLOOR_POSTURE.")

    y_pred = df_vis["alert_triggered"].astype(bool).astype(np.int64).to_numpy()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    far = float(false_alarm_rate(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["GT 0", "GT 1"])
    plt.ylabel("Ground truth (0=NORMAL, 1=ABNORMAL_FLOOR_POSTURE)")
    plt.xlabel("Prediction (alert_triggered)")
    plt.title("Confusion matrix (visibility filtered)")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG_PATH, dpi=150)
    plt.close()

    md = f"""# Metrics report

## Visibility filter
- `visibility_rate >= {thresh}`; excluded: {excluded_count} / {total_rows}.

## Labels
- GT NORMAL→0, ABNORMAL→1; pred: alert_triggered.

| Metric | Value |
|---:|---:|
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 | {f1:.4f} |
| FAR | {far:.4f} |

## Confusion matrix

```
{cm.tolist()}
```

`results/confusion_matrix.png`
"""

    METRICS_MD_PATH.write_text(md, encoding="utf-8")

    print("--- Metrics (visibility filtered, N=%d) ---" % len(df_vis))
    print(f"excluded_count (visibility < {thresh}): {excluded_count} / {total_rows}")
    print("| Метрика      | Значение |")
    print("|--------------|----------|")
    print(f"| Precision    | {precision:.4f} |")
    print(f"| Recall       | {recall:.4f} |")
    print(f"| F1-score     | {f1:.4f} |")
    print(f"| FAR          | {far:.4f} |")

if __name__ == "__main__":
    main()
