"""
Lightweight motion detector using frame differencing.

Downscales to 320px, converts to grayscale, computes absdiff with previous frame.
~0.3ms per call. Used to skip YOLO on static frames.
"""

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, threshold: float = 5.0, min_area_pct: float = 0.5,
                 warmup_frames: int = 5, resize_width: int = 320):
        self.threshold = threshold
        self.min_area_pct = min_area_pct / 100.0  # convert % to fraction
        self.warmup_frames = warmup_frames
        self.resize_width = resize_width

        self._prev_gray = None
        self._frame_count = 0

        # Stats
        self.total = 0
        self.motion = 0
        self.static = 0

    def has_motion(self, frame: np.ndarray) -> bool:
        """Check if frame has significant motion vs previous frame.

        Returns True during warmup period (first N frames).
        """
        self._frame_count += 1
        self.total += 1

        # Warmup: always return True for first N frames
        if self._frame_count <= self.warmup_frames:
            self._prev_gray = self._to_small_gray(frame)
            self.motion += 1
            return True

        small_gray = self._to_small_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = small_gray
            self.motion += 1
            return True

        # absdiff + threshold
        diff = cv2.absdiff(self._prev_gray, small_gray)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Fraction of pixels that changed
        changed_pct = np.count_nonzero(mask) / mask.size

        self._prev_gray = small_gray

        if changed_pct >= self.min_area_pct:
            self.motion += 1
            return True
        else:
            self.static += 1
            return False

    def _to_small_gray(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > self.resize_width:
            scale = self.resize_width / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def get_stats_str(self) -> str:
        skip_pct = (self.static / self.total * 100) if self.total > 0 else 0
        return (f"total:{self.total} motion:{self.motion} "
                f"static:{self.static} skip:{skip_pct:.0f}%")
