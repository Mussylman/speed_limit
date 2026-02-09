# file_logger.py
# Debug/performance event stream to JSONL files.
#
# Responsibilities (debug & perf trace):
#   - YOLO detections:   logs/detections.jsonl
#   - OCR attempts:      logs/ocr_attempts.jsonl
#   - System performance: logs/performance.jsonl
#
# NOT responsible for:
#   - Speed measurements → SpeedTracker (speed_tracker.py)
#   - Aggregate stats    → MetricsLogger  (metrics_logger.py)

import os
import json
import time
from typing import Dict, List
from threading import Lock


class JSONLWriter:
    """Thread-safe JSONL file writer."""

    def __init__(self, path: str):
        self.path = path
        self.lock = Lock()
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, data: dict):
        with self.lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


class FileLogger:
    """
    Debug/performance event stream.

    Files:
    - detections.jsonl:   YOLO detections per frame
    - ocr_attempts.jsonl: every OCR attempt with scores and reasons
    - performance.jsonl:  FPS, processing times, queue sizes
    """

    def __init__(self, output_dir: str):
        logs_dir = os.path.join(output_dir, "logs")

        self.detections = JSONLWriter(os.path.join(logs_dir, "detections.jsonl"))
        self.ocr_attempts = JSONLWriter(os.path.join(logs_dir, "ocr_attempts.jsonl"))
        self.performance = JSONLWriter(os.path.join(logs_dir, "performance.jsonl"))

        self._start_time = time.time()
        self._frame_count = 0
        self._last_perf_log = 0
        self._perf_interval = 1.0

        print(f"Logs: {logs_dir}/")

    def log_detections(
        self,
        frame_idx: int,
        detections: List[Dict],
        yolo_time_ms: float,
    ):
        """Log YOLO detections for a frame."""
        self.detections.write({
            "ts": time.time(),
            "frame": frame_idx,
            "count": len(detections),
            "yolo_ms": round(yolo_time_ms, 1),
            "objects": [
                {
                    "id": d.get("obj_id"),
                    "box": d.get("box"),
                    "conf": round(d.get("conf", 0), 3),
                }
                for d in detections
            ]
        })

    def log_ocr_attempt(
        self,
        frame_idx: int,
        track_id: int,
        status: str,  # "passed", "failed", "skipped"
        plate_text: str = "",
        car_score: float = 0.0,
        plate_score: float = 0.0,
        ocr_score: float = 0.0,
        blur: float = 0.0,
        brightness: float = 0.0,
        plate_width: int = 0,
        plate_height: int = 0,
        car_width: int = 0,
        car_height: int = 0,
        reason: str = "",
        processing_ms: float = 0.0,
    ):
        """Log an OCR attempt with scores and skip/fail reason."""
        self.ocr_attempts.write({
            "ts": time.time(),
            "frame": frame_idx,
            "track_id": track_id,
            "status": status,
            "plate": plate_text,
            "scores": {
                "car": round(car_score, 3),
                "plate": round(plate_score, 3),
                "ocr": round(ocr_score, 3),
                "total": round((car_score + plate_score + ocr_score) / 3, 3),
            },
            "raw": {
                "blur": round(blur, 1),
                "brightness": round(brightness, 1),
                "plate_px": [plate_width, plate_height],
                "car_px": [car_width, car_height],
            },
            "reason": reason,
            "proc_ms": round(processing_ms, 1),
        })

    def log_performance(
        self,
        frame_idx: int,
        fps: float,
        yolo_ms: float,
        ocr_queue: int = 0,
        yolo_queue: int = 0,
        cars_count: int = 0,
    ):
        """Log system performance metrics (throttled to once per second)."""
        now = time.time()
        if now - self._last_perf_log < self._perf_interval:
            return

        self._last_perf_log = now
        self.performance.write({
            "ts": now,
            "frame": frame_idx,
            "fps": round(fps, 1),
            "yolo_ms": round(yolo_ms, 1),
            "queues": {
                "ocr": ocr_queue,
                "yolo": yolo_queue,
            },
            "cars": cars_count,
            "uptime_s": round(now - self._start_time, 1),
        })
