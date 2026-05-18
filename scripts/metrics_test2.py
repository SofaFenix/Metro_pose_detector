"""test2 frame metrics with visibility filter."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

TEST2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TEST2_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

PREDICTIONS_CSV_PATH = TEST2_ROOT / "data" / "gt" / "predictions_test2.csv"
RESULTS_DIR = TEST2_ROOT / "results"
CONFUSION_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"
METRICS_MD_PATH = RESULTS_DIR / "metrics_report_test2.md"

def classify_gt_binary(series: pd.Series) -> np.ndarray:
    gt = np.full(len(series), fill_value=-1, dtype=np.int64)
    gt[(series == "NORMAL").to_numpy()] = 0
    gt[(series == "ABNORMAL_FLOOR_POSTURE").to_numpy()] = 1
    return gt

def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def main() -> None:
    if not PREDICTIONS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Отсутствует {PREDICTIONS_CSV_PATH}; зависимость: test2/scripts/pipeline_test2.py"
        )

    df = pd.read_csv(PREDICTIONS_CSV_PATH)
    total_rows = len(df)
    thresh = float(config.VISIBILITY_THRESHOLD)

    excluded_mask = df["visibility_rate"] < thresh
    excluded_count = int(excluded_mask.sum())
    df_vis = df.loc[~excluded_mask].reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if df_vis.empty:
        note = (
            f"# Metrics report (test2)\n\n**Нет строк после фильтра visibility >= {thresh}.**\n\n"
            f"| Исключено | {excluded_count} / {total_rows} |\n"
        )
        METRICS_MD_PATH.write_text(note, encoding="utf-8")
        print("Нет кадров с visibility_rate >= {:.2f}".format(thresh))
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
    plt.title("Confusion matrix test2 (visibility filtered)")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG_PATH, dpi=150)
    plt.close()

    md = f"""# Metrics report (test2)

## Visibility filter
- `visibility_rate >= {thresh}`; excluded: {excluded_count} / {total_rows}.

## Labels
- GT NORMAL→0, ABNORMAL_FLOOR_POSTURE→1; pred: alert_triggered.

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

`test2/results/confusion_matrix.png`
"""

    METRICS_MD_PATH.write_text(md, encoding="utf-8")

    print("--- Metrics test2 (visibility filtered, N=%d) ---" % len(df_vis))
    print(f"excluded_count (visibility < {thresh}): {excluded_count} / {total_rows}")
    print(f"| Precision | {precision:.4f} |")
    print(f"| Recall    | {recall:.4f} |")
    print(f"| F1        | {f1:.4f} |")
    print(f"| FAR       | {far:.4f} |")

if __name__ == "__main__":
    main()
