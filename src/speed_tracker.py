# speed_tracker.py
# Domain-level speed tracking: per-vehicle max speed, violation detection,
# raw measurement stream, final reports.
#
# Responsibilities (domain events):
#   - In-memory state: best (max) speed per track_id — used by UI at runtime
#   - Raw stream:      every measurement → speeds/measurements.jsonl
#   - Final reports:   speeds/all_speeds.json, speeds/violations.json
#
# NOT responsible for:
#   - Debug/perf trace (detections, OCR attempts, performance) → FileLogger
#   - Aggregate stats (FPS, sliding windows, periodic reports)  → MetricsLogger

import os
import json
import time
from datetime import datetime
from typing import Dict
from dataclasses import dataclass, asdict
from threading import Lock

import numpy as np


@dataclass
class SpeedEvent:
    """Speed data for a vehicle"""
    track_id: int = 0
    speed_kmh: float = 0.0
    timestamp: str = ""
    frame_idx: int = 0
    camera_id: str = ""

    speed_limit: int = 70
    is_violation: bool = False

    plate_text: str = ""
    plate_conf: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, (np.bool_, bool)):
                d[key] = bool(value)
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                d[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                d[key] = float(value)
        return d


class SpeedTracker:
    """
    Domain-level speed tracker.

    Tracks max speed per vehicle, detects violations, streams raw
    measurements to JSONL, and writes final summary reports.

    Output:
        output_dir/speeds/
            measurements.jsonl   — every speed measurement (streaming)
            all_speeds.json      — final: all vehicles with max speed
            violations.json      — final: only violators
    """

    def __init__(
        self,
        output_dir: str,
        camera_id: str = "camera_01",
        speed_limit: int = 70,
        video_start_time: "datetime | None" = None,
        video_fps: float = 25.0,
    ):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.speed_limit = speed_limit
        self.video_start_time = video_start_time  # None = use system time
        self.video_fps = video_fps

        self.speeds_dir = os.path.join(output_dir, "speeds")
        os.makedirs(self.speeds_dir, exist_ok=True)

        # Best (max) speed per track_id — used at runtime by UI
        self.speeds: Dict[int, SpeedEvent] = {}

        # Stats
        self.stats = {
            "total_measurements": 0,
            "unique_vehicles": 0,
            "violations": 0,
            "max_speed": 0.0,
            "avg_speed": 0.0,
        }

        # Raw measurement stream (JSONL)
        self._jsonl_path = os.path.join(self.speeds_dir, "measurements.jsonl")
        self._jsonl_lock = Lock()

        print(f"Speed limit: {speed_limit} km/h")

    def _frame_timestamp(self, frame_idx: int) -> str:
        """Compute timestamp for a frame. Uses video file time if available."""
        if self.video_start_time is not None and frame_idx >= 0:
            from datetime import timedelta
            dt = self.video_start_time + timedelta(seconds=frame_idx / self.video_fps)
            return dt.isoformat()
        return datetime.now().isoformat()

    def update(
        self,
        track_id: int,
        speed_kmh: float,
        frame_idx: int = 0,
        plate_text: str = "",
        plate_conf: float = 0.0,
    ):
        """Update speed for track_id (keeps max). Also streams raw measurement."""
        if speed_kmh <= 0:
            return

        self.stats["total_measurements"] += 1

        # Stream raw measurement to JSONL
        self._write_measurement(frame_idx, track_id, speed_kmh, plate_text)

        current = self.speeds.get(track_id)

        # Keep max speed
        if current is None or speed_kmh > current.speed_kmh:
            event = SpeedEvent(
                track_id=track_id,
                speed_kmh=round(speed_kmh, 1),
                timestamp=self._frame_timestamp(frame_idx),
                frame_idx=frame_idx,
                camera_id=self.camera_id,
                speed_limit=self.speed_limit,
                is_violation=speed_kmh > self.speed_limit,
                plate_text=plate_text,
                plate_conf=round(plate_conf, 2) if plate_conf else 0.0,
            )
            self.speeds[track_id] = event

            if event.is_violation and (current is None or not current.is_violation):
                print(f"VIOLATION! ID:{track_id} -> {speed_kmh:.0f} km/h (limit {self.speed_limit})")

    def update_plate(self, track_id: int, plate_text: str, plate_conf: float):
        """Update plate for an existing speed record"""
        if track_id in self.speeds:
            self.speeds[track_id].plate_text = plate_text
            self.speeds[track_id].plate_conf = round(plate_conf, 2)

    def _write_measurement(self, frame_idx: int, track_id: int,
                           speed_kmh: float, plate_text: str):
        """Stream a single measurement to JSONL."""
        record = {
            "ts": time.time(),
            "frame": frame_idx,
            "track_id": track_id,
            "speed": round(speed_kmh, 1),
            "plate": plate_text,
        }
        with self._jsonl_lock:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finalize(self):
        """Save final reports to disk."""
        all_speeds = []
        violations = []

        speeds_list = list(self.speeds.values())

        for event in speeds_list:
            data = event.to_dict()
            all_speeds.append(data)
            if event.is_violation:
                violations.append(data)

        if speeds_list:
            self.stats["unique_vehicles"] = len(speeds_list)
            self.stats["violations"] = len(violations)
            self.stats["max_speed"] = max(e.speed_kmh for e in speeds_list)
            self.stats["avg_speed"] = round(
                sum(e.speed_kmh for e in speeds_list) / len(speeds_list), 1)

        # All speeds
        all_path = os.path.join(self.speeds_dir, "all_speeds.json")
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_id": self.camera_id,
                "speed_limit": self.speed_limit,
                "stats": self.stats,
                "vehicles": all_speeds,
            }, f, ensure_ascii=False, indent=2)

        # Violations only
        viol_path = os.path.join(self.speeds_dir, "violations.json")
        with open(viol_path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_id": self.camera_id,
                "speed_limit": self.speed_limit,
                "total_violations": len(violations),
                "vehicles": violations,
            }, f, ensure_ascii=False, indent=2)

        print(f"\nSpeed stats:")
        print(f"   Total measurements: {self.stats['total_measurements']}")
        print(f"   Unique vehicles: {self.stats['unique_vehicles']}")
        print(f"   Average speed: {self.stats['avg_speed']} km/h")
        print(f"   Max speed: {self.stats['max_speed']} km/h")
        print(f"   Violations (>{self.speed_limit} km/h): {self.stats['violations']}")
        print(f"\nSpeeds dir: {self.speeds_dir}")
