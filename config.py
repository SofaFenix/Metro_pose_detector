MODEL_VERSIONS = ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolo11n-pose.pt", "yolo11s-pose.pt"]
# exp1_detector_eval_fix
DEFAULT_MODEL = "yolov8n-pose.pt"
# Grid Search, exp B
THETA_THRESHOLD = 9
K_BASE = 0.777
K_CRITICAL = 0.977
WINDOW_SIZE = 10  # temporal window, frames
MIN_ALERT_FRAMES = 4  # abnormal count in window
VISIBILITY_THRESHOLD = 0.7  # visible keypoints fraction
CONFIDENCE_THRESHOLD = 0.5  # keypoint confidence
ROI_MASK_PATH = None  # ROI mask path; None = full frame
FALLBACK_K = 5  # occlusion hold frames

# MS temporal filter; time_tune Grid Search (test3_temporal_tuning/results/selected_params.md)
TEMPORAL_WINDOW_MS = 600  # window length, ms
MIN_ALERT_RATIO = 0.7  # is_abnormal_raw ratio in window

# Grid Search ranges, exp A/B
GRID_THETA_MIN = 9
GRID_THETA_MAX = 119
GRID_THETA_STEP = 1
GRID_K_BASE_MIN = 0.727
GRID_K_BASE_MAX = 1.102
GRID_K_BASE_STEP = 0.05
GRID_K_CRITICAL_MIN = 0.927
GRID_K_CRITICAL_MAX = 3.772
GRID_K_CRITICAL_STEP = 0.05
