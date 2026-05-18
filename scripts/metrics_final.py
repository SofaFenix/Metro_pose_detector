"""Frame metrics P/R/F1/FAR; visibility >= config.VISIBILITY_THRESHOLD."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_TEST_ROOT = PROJECT_ROOT.parent
_REPO_ROOT = FINAL_TEST_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

PRED_PATH = FINAL_TEST_ROOT / "data" / "predictions_final.csv"
RESULTS_DIR = FINAL_TEST_ROOT / "results"
CONFUSION_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"
METRICS_MD_PATH = RESULTS_DIR / "metrics_report.md"

def classify_gt_binary(series: pd.Series) -> np.ndarray:
    gt = np.full(len(series), fill_value=-1, dtype=np.int64)
    s = series.astype(str).str.strip()
    gt[(s == "NORMAL").to_numpy()] = 0
    mask_abn = s.str.upper().str.contains("ABNORMAL", na=False)
    gt[mask_abn.to_numpy()] = 1
    return gt

def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Отсутствует {PRED_PATH}; зависимость: final_test/scripts/pipeline_final.py")

    df = pd.read_csv(PRED_PATH)
    total_rows = len(df)
    thresh = float(config.VISIBILITY_THRESHOLD)

    excluded_mask = df["visibility_rate"] < thresh
    excluded_count = int(excluded_mask.sum())
    df_vis = df.loc[~excluded_mask].reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if df_vis.empty:
        note = (
            f"# Metrics report (final_test)\n\n**Нет строк после фильтра visibility >= {thresh}.**\n\n"
            f"| Исключено | {excluded_count} / {total_rows} |\n"
        )
        METRICS_MD_PATH.write_text(note, encoding="utf-8")
        print("Нет кадров с visibility_rate >= {:.2f}".format(thresh))
        return

    y_true = classify_gt_binary(df_vis["gt_class"])
    if np.any(y_true < 0):
        raise ValueError("В GT встречаются классы, не сводимые к NORMAL / ABNORMAL.")

    y_pred = df_vis["alert_triggered"].astype(int).clip(0, 1).to_numpy()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["GT 0", "GT 1"],
    )
    plt.ylabel("Ground truth (0=NORMAL, 1=ABNORMAL)")
    plt.xlabel("Prediction (alert_triggered)")
    plt.title("Confusion matrix final_test (visibility filtered)")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG_PATH, dpi=150)
    plt.close()

    md = f"""# Metrics report (final_test)

## Visibility filter
- `visibility_rate >= {thresh}`; excluded: {excluded_count} / {total_rows}.

## Labels
- GT NORMAL→0; ABNORMAL→1; pred: alert_triggered. FAR = FP/(FP+TN).

## Metrics

| Метрика | Значение |
|---:|---:|
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-score | {f1:.4f} |
| FAR | {far:.4f} |

## Confusion matrix

Изображение: `final_test/results/confusion_matrix.png`

```
{cm.tolist()}
```
"""

    METRICS_MD_PATH.write_text(md, encoding="utf-8")

    print("--- Metrics final_test (visibility filtered, N=%d) ---" % len(df_vis))
    print(f"excluded_count (visibility < {thresh}): {excluded_count} / {total_rows}")
    print(f"| Precision | {precision:.4f} |")
    print(f"| Recall    | {recall:.4f} |")
    print(f"| F1        | {f1:.4f} |")
    print(f"| FAR       | {far:.4f} |")

if __name__ == "__main__":
    main()
