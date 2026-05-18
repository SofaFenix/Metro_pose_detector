# Metro Pose Anomaly Detection (Zero-shot Geometry)

## Назначение
Детекция нештатных поз на уровне пола платформы (горизонтальная поза) в видеопотоке CCTV.

## Архитектура
YOLOv8/v11-Pose → геометрические признаки (θ, K_ar) → rule-based классификатор → временной фильтр → fallback при окклюзии.

## Модель
`yolov8n-pose.pt` (zero-shot). Метрики выбора: PCK@0.1=0.9445, FPS=6.82 (`results/metrics_pose_comparison.md`).

## Зависимости
Python 3.10+, `requirements.txt`, Ultralytics YOLO-Pose.

## Конфигурация
Пороги и параметры: `config.py`. Артефакты экспериментов: `checkpoints/`.
