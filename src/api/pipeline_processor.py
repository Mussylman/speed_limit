# pipeline_processor.py
# Frame processor for MJPEG stream: AsyncYOLO + speed + AsyncOCR

import os
import cv2
import time
import datetime
import numpy as np
from collections import deque
from threading import Lock
from typing import Dict, List

from config import BASE_DIR
from pipeline_builder import create_yolo, create_speed_estimator, create_ocr


class PipelineProcessor:
    """
    Frame processor: AsyncYOLO + tracking + speed + AsyncOCR.
    YOLO and OCR run in separate threads — non-blocking for the stream.
    """

    def __init__(self, config: dict, cam_config: dict, camera_id: str, source_fps: float = 0):
        self.config = config
        self.cam_config = cam_config
        self.camera_id = camera_id
        self.source_fps = source_fps
        self.frame_idx = 0
        self.frame_skip = int(config.get("frame_skip", 1))
        self.speed_limit = int(config.get("speed_limit", 70))

        # Cached detections for smooth display
        self.cached_detections: List[dict] = []
        self._lock = Lock()

        # Statistics
        self.stats_total_cars = 0
        self.stats_total_detections = 0
        self.stats_plates_recognized = 0
        self.stats_violations = 0
        self._seen_track_ids: set = set()
        self._recognized_track_ids: set = set()
        self._violation_track_ids: set = set()

        # Recent results for panel
        self._recent_results: List[dict] = []
        self._max_recent = 12

        # FPS tracking
        self._fps = 0.0
        self._fps_frames = 0
        self._fps_start = time.time()

        # Latency tracking
        self._latency_ms = 0.0
        self._latency_p95 = 0.0
        self._latency_history: deque = deque(maxlen=300)
        self._latency_calc_time = time.time()

        # Init pipeline via shared builder
        fps = self.source_fps if self.source_fps > 0 else config.get("fps", 30)
        print(f"[Pipeline] Speed FPS: {fps}")

        self.async_yolo, use_half, self.yolo_imgsz = create_yolo(config)
        self.speed, self.speed_method = create_speed_estimator(
            config, cam_config, camera_id, fps)

        # OCR with output directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(BASE_DIR, config["output_dir"],
                               f"{camera_id}_run_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)

        try:
            self.recognizer, self.async_ocr = create_ocr(config, camera_id, out_dir)
            print(f"[Pipeline] OCR: NomeroffNet (async, output={out_dir})")
        except Exception as e:
            self.recognizer = None
            self.async_ocr = None
            print(f"[Pipeline] OCR: disabled ({e})")

        print(f"[Pipeline] {camera_id}: initialized (async, skip={self.frame_skip})")

    def process_frame(self, frame: np.ndarray, capture_ts: float = 0.0) -> np.ndarray:
        """Process a single frame — called by MJPEG streamer (non-blocking)"""
        self.frame_idx += 1

        # FPS counter
        self._fps_frames += 1
        now = time.time()
        if now - self._fps_start >= 1.0:
            self._fps = self._fps_frames / (now - self._fps_start)
            self._fps_frames = 0
            self._fps_start = now

        # Latency tracking
        if capture_ts > 0:
            self._latency_ms = (now - capture_ts) * 1000
            self._latency_history.append(self._latency_ms)
            if now - self._latency_calc_time >= 1.0 and len(self._latency_history) > 10:
                sorted_lat = sorted(self._latency_history)
                self._latency_p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
                self._latency_calc_time = now

        # Submit to AsyncYOLO (non-blocking, every Nth frame)
        if self.frame_skip <= 1 or self.frame_idx % self.frame_skip == 0:
            self.async_yolo.submit(frame, self.frame_idx)

        # Collect ready YOLO results (non-blocking)
        self._collect_results(frame)

        # Collect OCR results (non-blocking)
        if self.async_ocr:
            for ocr_result in self.async_ocr.get_results(current_frame_idx=self.frame_idx):
                pass  # results are cached inside async_ocr

        # Draw results on frame
        return self._draw_frame(frame)

    def _collect_results(self, frame: np.ndarray):
        """Collect results from AsyncYOLO (non-blocking)"""
        for yolo_result in self.async_yolo.get_results():
            new_detections = []

            for det in yolo_result.detections:
                obj_id = det.obj_id
                x1, y1, x2, y2 = det.box

                # Speed
                self.speed.process(yolo_result.frame_idx, obj_id, det.cx, det.cy)

                current_speed = None
                if self.speed_method == "homography":
                    v = self.speed.get_speed(obj_id)
                    if v:
                        current_speed = v
                else:
                    if hasattr(self.speed, 'object_speeds') and obj_id in self.speed.object_speeds:
                        v, _ = self.speed.object_speeds[obj_id]
                        current_speed = v

                # Submit OCR (non-blocking)
                if self.async_ocr:
                    self.async_ocr.submit(
                        track_id=obj_id,
                        crop=det.crop,
                        car_height=y2 - y1,
                        car_width=x2 - x1,
                        frame_idx=yolo_result.frame_idx,
                        detection_conf=det.conf,
                    )

                # Get filtered OCR result (only passed_results — validated plates)
                plate_text = ""
                plate_conf = 0.0
                if self.recognizer and hasattr(self.recognizer, 'passed_results'):
                    if obj_id in self.recognizer.passed_results:
                        pr = self.recognizer.passed_results[obj_id]
                        plate_text = pr.plate_text or ""
                        plate_conf = pr.ocr_conf or 0.0

                new_detections.append({
                    'box': det.box,
                    'obj_id': obj_id,
                    'conf': det.conf,
                    'plate_text': plate_text,
                    'plate_conf': plate_conf,
                    'speed': current_speed,
                })

                # --- Statistics ---
                self.stats_total_detections += 1

                if obj_id not in self._seen_track_ids:
                    self._seen_track_ids.add(obj_id)
                    self.stats_total_cars += 1

                if plate_text and obj_id not in self._recognized_track_ids:
                    self._recognized_track_ids.add(obj_id)
                    self.stats_plates_recognized += 1
                    self._recent_results.insert(0, {
                        'plate': plate_text,
                        'conf': plate_conf,
                        'speed': current_speed,
                        'time': time.strftime("%H:%M:%S"),
                        'violation': current_speed is not None and current_speed > self.speed_limit,
                    })
                    if len(self._recent_results) > self._max_recent:
                        self._recent_results = self._recent_results[:self._max_recent]

                if current_speed is not None and current_speed > self.speed_limit:
                    if obj_id not in self._violation_track_ids:
                        self._violation_track_ids.add(obj_id)
                        self.stats_violations += 1

            # Cleanup old tracks
            if self.speed_method == "homography":
                self.speed.cleanup_old_tracks(yolo_result.frame_idx)

            with self._lock:
                self.cached_detections = new_detections

    def _draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results on frame + right panel with stats and plates"""
        h, w = frame.shape[:2]
        panel_w = 320

        # Create canvas: video + right panel
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        # Draw speed zones on video area
        if self.speed_method == "lines" and hasattr(self.speed, 'draw_zones'):
            self.speed.draw_zones(canvas[:, :w])

        # Draw detections on video area
        with self._lock:
            detections = list(self.cached_detections)

        for det in detections:
            x1, y1, x2, y2 = det['box']
            obj_id = det['obj_id']
            plate_text = det['plate_text']
            plate_conf = det['plate_conf']
            speed = det['speed']

            box_color = (0, 255, 0) if plate_text else (0, 255, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            label = f"ID:{obj_id}"
            cv2.putText(canvas, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if speed:
                color = (0, 255, 0) if speed <= self.speed_limit else (0, 0, 255)
                cv2.putText(canvas, f"{speed:.0f} km/h", (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if plate_text:
                cv2.putText(canvas, f"{plate_text} ({plate_conf:.2f})",
                            (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Camera name on video
        cv2.putText(canvas, self.camera_id, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ========== RIGHT PANEL ==========
        px = w + 10
        panel_bg = canvas[:, w:, :]
        panel_bg[:] = (30, 30, 30)

        cv2.line(canvas, (w, 0), (w, h), (80, 80, 80), 2)

        # --- Header ---
        cy = 30
        cv2.putText(canvas, "SPEED MONITOR", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cy += 30
        cv2.line(canvas, (w + 5, cy - 5), (w + panel_w - 5, cy - 5), (80, 80, 80), 1)

        # --- Stats ---
        cy += 25
        cv2.putText(canvas, f"FPS: {self._fps:.1f}", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(canvas, f"Now: {len(detections)} cars", (px + 150, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Latency
        cy += 22
        lat_color = (0, 255, 0) if self._latency_ms < 100 else (0, 200, 255) if self._latency_ms < 200 else (0, 0, 255)
        cv2.putText(canvas, f"Latency: {self._latency_ms:.0f}ms  p95: {self._latency_p95:.0f}ms", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lat_color, 1)

        cy += 25
        cv2.putText(canvas, f"Total cars:", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, f"{self.stats_total_cars}", (px + 180, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cy += 25
        cv2.putText(canvas, f"Plates recognized:", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, f"{self.stats_plates_recognized}", (px + 180, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cy += 25
        cv2.putText(canvas, f"Violations:", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, f"{self.stats_violations}", (px + 180, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Recognition rate
        cy += 25
        rate = (self.stats_plates_recognized / self.stats_total_cars * 100) if self.stats_total_cars > 0 else 0
        cv2.putText(canvas, f"OCR rate:", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        rate_color = (0, 255, 0) if rate > 50 else (0, 200, 255) if rate > 20 else (0, 0, 255)
        cv2.putText(canvas, f"{rate:.1f}%", (px + 180, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2)

        cy += 20
        cv2.line(canvas, (w + 5, cy), (w + panel_w - 5, cy), (80, 80, 80), 1)

        # --- Recent results header ---
        cy += 25
        cv2.putText(canvas, "RECENT PLATES", (px, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cy += 10

        # --- Recent results list ---
        row_h = 45
        for i, r in enumerate(self._recent_results):
            if cy + row_h > h - 10:
                break

            cy += 5
            row_color = (40, 20, 20) if r.get('violation') else (20, 40, 20)
            cv2.rectangle(canvas, (w + 5, cy), (w + panel_w - 5, cy + row_h - 5),
                          row_color, -1)
            cv2.rectangle(canvas, (w + 5, cy), (w + panel_w - 5, cy + row_h - 5),
                          (60, 60, 60), 1)

            plate_color = (100, 100, 255) if r.get('violation') else (100, 255, 100)
            cv2.putText(canvas, r['plate'], (px, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, plate_color, 2)

            spd_text = f"{r['speed']:.0f} km/h" if r.get('speed') else "-- km/h"
            spd_color = (0, 0, 255) if r.get('violation') else (200, 200, 200)
            cv2.putText(canvas, spd_text, (px, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, spd_color, 1)

            cv2.putText(canvas, r['time'], (px + 180, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

            cv2.putText(canvas, f"{r['conf']:.0%}", (px + 230, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

            cy += row_h

        # Speed limit badge on video
        cv2.putText(canvas, f"Limit: {self.speed_limit} km/h", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        return canvas

    def stop(self):
        """Clean shutdown"""
        passed = self.stats_plates_recognized
        total = self.stats_total_cars
        pct = (passed / total * 100) if total > 0 else 0
        print(f"[Pipeline] {self.camera_id}: Passed: {passed}/{total} ({pct:.1f}%)")

        if hasattr(self, 'async_yolo'):
            self.async_yolo.stop()
        if self.async_ocr:
            self.async_ocr.stop()


# Global pipeline instances
_pipelines: Dict[str, PipelineProcessor] = {}
_pipelines_lock = Lock()


def get_or_create_pipeline(config: dict, cam_config: dict, camera_id: str, source_fps: float = 0) -> PipelineProcessor:
    with _pipelines_lock:
        if camera_id not in _pipelines:
            _pipelines[camera_id] = PipelineProcessor(config, cam_config, camera_id, source_fps=source_fps)
        return _pipelines[camera_id]


def stop_pipeline(camera_id: str):
    with _pipelines_lock:
        if camera_id in _pipelines:
            _pipelines[camera_id].stop()
            del _pipelines[camera_id]
