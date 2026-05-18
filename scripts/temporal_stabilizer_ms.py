"""MS temporal filter and occlusion fallback; VISIBILITY_THRESHOLD from config."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

STATUS_ABNORMAL = "ABNORMAL_HORIZONTAL_POSTURE"
STATUS_NORMAL = "NORMAL"
STATUS_UNRELIABLE = "UNRELIABLE"

class TemporalStabilizerMs:
    def __init__(self, window_ms: float, min_alert_ratio: float, fallback_k: int) -> None:
        self.window_ms = float(window_ms)
        self.min_alert_ratio = float(min_alert_ratio)
        self.fallback_k = int(fallback_k)
        self.deque: collections.deque[tuple[float, bool]] = collections.deque()
        self.last_valid_state: str | None = None
        self.occlusion_counter = 0

    def update(self, current_ts_sec: float, is_abn_raw: bool, visibility: float) -> str:
        thresh = float(config.VISIBILITY_THRESHOLD)
        if visibility < thresh:
            self.occlusion_counter += 1
            if self.occlusion_counter > self.fallback_k:
                return STATUS_UNRELIABLE
            if self.last_valid_state is not None:
                return self.last_valid_state
            return STATUS_NORMAL

        self.occlusion_counter = 0
        self.deque.append((float(current_ts_sec), bool(is_abn_raw)))
        window_sec = self.window_ms / 1000.0
        while self.deque and current_ts_sec - self.deque[0][0] > window_sec:
            self.deque.popleft()

        if len(self.deque) > 0:
            ratio = sum(1 for _, v in self.deque if v) / len(self.deque)
            if ratio >= self.min_alert_ratio:
                self.last_valid_state = STATUS_ABNORMAL
            else:
                self.last_valid_state = STATUS_NORMAL

        if self.last_valid_state is None:
            return STATUS_NORMAL
        return self.last_valid_state
