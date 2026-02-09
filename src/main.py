# main.py
# Pipeline: YOLO tracking + speed + OCR (NomeroffNet)
# OCR every frame, best confidence result is kept

import os
import cv2
import time
import datetime
import numpy as np
from typing import Tuple
from threading import Thread
from queue import Queue

from config import BASE_DIR, HEADLESS
from pipeline_builder import create_yolo, create_speed_estimator, create_ocr
from speed_tracker import SpeedTracker
from metrics_logger import MetricsLogger
from file_logger import FileLogger


class AsyncVideoWriter:
    """Async video writing in a separate thread"""
    def __init__(self, path: str, fourcc: int, fps: float, size: Tuple[int, int]):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.queue: Queue = Queue(maxsize=30)
        self.running = True
        self.thread = Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
            except:
                pass

    def write(self, frame: np.ndarray):
        if not self.queue.full():
            self.queue.put(frame.copy())

    def release(self):
        self.running = False
        self.thread.join(timeout=5.0)
        self.writer.release()


def draw_results_panel(frame, passed_results, speed_tracker, speed_limit):
    """Draw panel with plates and speeds on the left"""
    h, w = frame.shape[:2]
    panel_width = 280

    cv2.rectangle(frame, (0, 0), (panel_width, h), (30, 30, 30), -1)

    cv2.putText(frame, "PASSED VEHICLES", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(frame, (10, 50), (panel_width - 10, 50), (100, 100, 100), 1)

    y_offset = 80
    items = list(passed_results.items())[-10:]

    for track_id, event in reversed(items):
        plate = event.plate_text

        speed_event = speed_tracker.speeds.get(track_id)
        spd = speed_event.speed_kmh if speed_event else 0

        if spd > speed_limit:
            speed_color = (0, 0, 255)
            status = "!"
        else:
            speed_color = (0, 255, 0)
            status = ""

        cv2.putText(frame, f"{plate}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        speed_text = f"{spd:.0f} km/h {status}"
        cv2.putText(frame, speed_text, (15, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)

        y_offset += 60

        if y_offset > h - 50:
            break

    total = len(passed_results)
    violations = sum(1 for tid in passed_results
                     if tid in speed_tracker.speeds
                     and speed_tracker.speeds[tid].speed_kmh > speed_limit)

    cv2.putText(frame, f"Total: {total}  Violations: {violations}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def process_source(source, cfg, cam_cfg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = source.info.name
    out_dir = os.path.join(BASE_DIR, cfg["output_dir"], f"{name}_run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nStart: {name}")
    print(f"Output: {out_dir}")
    print(f"Backend: {source.info.backend}")

    # --- Pipeline components via builder ---
    async_yolo, use_half, yolo_imgsz = create_yolo(cfg)

    if use_half:
        print("YOLO: FP16 mode enabled")

    fps = source.info.fps if source.info.fps > 0 else cfg.get("fps", 30)

    speed, speed_method = create_speed_estimator(cfg, cam_cfg, name, fps, source=source)
    show_bird_eye = False
    if speed_method == "homography":
        show_bird_eye = cfg.get("homography", {}).get("show_bird_eye", True)
        print("Speed: HOMOGRAPHY")
    else:
        print("Speed: LINES")

    recognizer, async_ocr = create_ocr(cfg, name, out_dir)

    plate_format = cfg.get("plate_format_regex", "")
    print(f"OCR: NomeroffNet (ASYNC)")
    print(f"   Car: >= {cfg.get('min_car_height', 150)}x{cfg.get('min_car_width', 100)} px")
    print(f"   Plate: >= {cfg.get('min_plate_width', 60)}x{cfg.get('min_plate_height', 15)} px")
    print(f"   Format: {plate_format if plate_format else 'any'}")
    print(f"   Cooldown: {cfg.get('ocr_cooldown_frames', 3)} frames")

    # Parameters
    ocr_conf_threshold = float(cfg.get("ocr_conf_threshold", 0.5))
    frame_skip = int(cfg.get("frame_skip", 1))
    show_window = bool(cfg.get("show_window", True)) and not HEADLESS
    speed_limit = int(cfg.get("speed_limit", 70))

    print(f"YOLO: imgsz={yolo_imgsz}, skip={frame_skip}")

    # Speed tracker (domain: per-vehicle max speed, violations, raw stream)
    speed_tracker = SpeedTracker(
        output_dir=out_dir,
        camera_id=name,
        speed_limit=speed_limit,
    )

    # Metrics
    metrics = MetricsLogger(
        output_dir=out_dir,
        report_interval=10.0,
        json_log=True,
    )
    metrics.async_ocr = async_ocr

    # File logger
    file_logger = FileLogger(output_dir=out_dir)
    recognizer.file_logger = file_logger

    print(f"YOLO: async mode (frame_skip={frame_skip})")
    print("Running... (Q to quit)\n")

    # Video recording
    records_dir = os.path.join(BASE_DIR, "records")
    os.makedirs(records_dir, exist_ok=True)
    video_out_path = os.path.join(records_dir, f"{name}_{timestamp}.mp4")
    video_writer = None

    frame_idx = 0

    # Cached detections for smooth video
    cached_detections = []
    last_yolo_frame = 0
    last_yolo_time_ms = 0

    # Display FPS control
    target_frame_time = 1.0 / fps
    last_display_time = time.time()

    # Display FPS counter
    display_fps = 0.0
    display_fps_frames = 0
    display_fps_start = time.time()

    # Latency tracking
    from collections import deque
    latency_history = deque(maxlen=300)
    latency_ms = 0.0
    latency_p95 = 0.0
    latency_calc_time = time.time()

    while True:
        frame_start_time = time.time()

        # Display FPS
        display_fps_frames += 1
        now_fps = time.time()
        if now_fps - display_fps_start >= 1.0:
            display_fps = display_fps_frames / (now_fps - display_fps_start)
            display_fps_frames = 0
            display_fps_start = now_fps

        ok, frame, capture_ts = source.read()

        if not ok or frame is None:
            if show_window:
                cv2.imshow(name, np.zeros((360, 640, 3), dtype=np.uint8))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # Latency: capture -> now
        if capture_ts > 0:
            latency_ms = (time.time() - capture_ts) * 1000
            latency_history.append(latency_ms)
            now_lat = time.time()
            if now_lat - latency_calc_time >= 1.0 and len(latency_history) > 10:
                sorted_lat = sorted(latency_history)
                latency_p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
                latency_calc_time = now_lat

        frame_idx += 1

        # === SUBMIT TO YOLO (async, every Nth frame) ===
        if frame_skip <= 1 or frame_idx % frame_skip == 0:
            async_yolo.submit(frame, frame_idx)

        # === GET YOLO RESULTS (non-blocking) ===
        for yolo_result in async_yolo.get_results():
            metrics.start_frame(yolo_result.frame_idx)
            last_yolo_time_ms = yolo_result.processing_time_ms
            last_yolo_frame = yolo_result.frame_idx

            cached_detections = []

            detections = yolo_result.detections
            confs_list = [d.conf for d in detections]
            metrics.log_yolo(yolo_result.processing_time_ms, len(detections), confs_list)

            file_logger.log_detections(
                frame_idx=yolo_result.frame_idx,
                detections=[{"obj_id": d.obj_id, "box": d.box, "conf": d.conf} for d in detections],
                yolo_time_ms=yolo_result.processing_time_ms,
            )

            for det in detections:
                obj_id = det.obj_id
                x1i, y1i, x2i, y2i = det.box

                speed.process(yolo_result.frame_idx, obj_id, det.cx, det.cy)

                async_ocr.submit(
                    track_id=obj_id,
                    crop=det.crop,
                    car_height=y2i - y1i,
                    car_width=x2i - x1i,
                    frame_idx=yolo_result.frame_idx,
                    detection_conf=det.conf,
                )

                cached_result = async_ocr.get_cached_result(obj_id)
                plate_text = cached_result.plate_text if cached_result else ""
                plate_conf = cached_result.ocr_conf if cached_result else 0.0

                current_speed = None
                if speed_method == "homography":
                    v = speed.get_speed(obj_id)
                    if v:
                        current_speed = v
                else:
                    if hasattr(speed, 'object_speeds') and obj_id in speed.object_speeds:
                        v, _ = speed.object_speeds[obj_id]
                        current_speed = v

                if current_speed:
                    is_violation = current_speed > speed_limit
                    speed_tracker.update(
                        track_id=obj_id,
                        speed_kmh=current_speed,
                        frame_idx=yolo_result.frame_idx,
                        plate_text=plate_text,
                        plate_conf=plate_conf,
                    )
                    metrics.log_speed(current_speed, is_violation)

                cached_detections.append({
                    'box': det.box,
                    'obj_id': obj_id,
                    'plate_text': plate_text,
                    'plate_conf': plate_conf,
                    'speed': current_speed,
                })

            if speed_method == "homography":
                speed.cleanup_old_tracks(yolo_result.frame_idx)

            file_logger.log_performance(
                frame_idx=yolo_result.frame_idx,
                fps=metrics.current_fps,
                yolo_ms=yolo_result.processing_time_ms,
                ocr_queue=async_ocr.queue_size(),
                yolo_queue=async_yolo.input_queue.qsize(),
                cars_count=len(detections),
            )

            metrics.update_skip_reasons(recognizer.stats)
            metrics.end_frame()

        # === GET OCR RESULTS ===
        for ocr_result in async_ocr.get_results(current_frame_idx=frame_idx):
            if ocr_result.blur_score > 0:
                metrics.log_quality(ocr_result.blur_score, ocr_result.brightness_score)
            if ocr_result.result and ocr_result.result.plate_text:
                metrics.log_ocr(
                    called=True,
                    time_ms=ocr_result.processing_time_ms,
                    result=ocr_result.result.plate_text,
                    conf=ocr_result.result.ocr_conf,
                    passed=True,
                )

        # === DRAWING (every frame) ===
        raw_frame = frame.copy()

        if speed_method == "lines" and hasattr(speed, 'draw_zones'):
            speed.draw_zones(raw_frame)

        for det in cached_detections:
            x1i, y1i, x2i, y2i = det['box']
            obj_id = det['obj_id']
            plate_text = det['plate_text']
            plate_conf = det['plate_conf']
            current_speed = det['speed']

            box_color = (0, 255, 0) if plate_conf >= ocr_conf_threshold else (0, 255, 255)
            cv2.rectangle(raw_frame, (x1i, y1i), (x2i, y2i), box_color, 2)
            cv2.putText(raw_frame, f"ID:{obj_id}", (x1i, y1i - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if plate_text:
                cv2.putText(raw_frame, f"{plate_text} ({plate_conf:.2f})",
                            (x1i, y2i + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if current_speed:
                c = (0, 255, 0) if current_speed < speed_limit else (0, 0, 255)
                cv2.putText(raw_frame, f"{current_speed:.0f} km/h", (x1i, y1i - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

        if show_window:
            if recognizer.passed_results:
                raw_frame = draw_results_panel(
                    raw_frame, recognizer.passed_results,
                    speed_tracker, speed_limit
                )

            if show_bird_eye and speed_method == "homography":
                if last_yolo_frame == frame_idx or not hasattr(process_source, '_bev_cache'):
                    process_source._bev_cache = speed.draw_bird_eye_view(size=(300, 400), frame_idx=frame_idx)
                bev = process_source._bev_cache
                bev_h, bev_w = bev.shape[:2]
                raw_frame[10:10+bev_h, raw_frame.shape[1]-bev_w-10:raw_frame.shape[1]-10] = bev

            mx = 290
            cv2.putText(raw_frame, f"FPS: {display_fps:.1f} | YOLO: {metrics.current_fps:.1f}", (mx, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            dec_fps = source.decode_fps if hasattr(source, 'decode_fps') else 0
            q_sz = source.queue_size if hasattr(source, 'queue_size') else 0
            q_cap = source.queue_capacity if hasattr(source, 'queue_capacity') else 0
            dropped = source.frames_dropped if hasattr(source, 'frames_dropped') else 0
            cv2.putText(raw_frame, f"Decode: {dec_fps:.0f} | Q: {q_sz}/{q_cap} | Drop: {dropped}", (mx, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            lat_color = (0, 255, 0) if latency_ms < 100 else (0, 200, 255) if latency_ms < 200 else (0, 0, 255)
            cv2.putText(raw_frame, f"Latency: {latency_ms:.0f}ms (p95: {latency_p95:.0f}ms)", (mx, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, lat_color, 1)

            # Passed rate (live)
            total_v = len(speed_tracker.speeds)
            passed_v = len(recognizer.passed_results)
            passed_pct = (passed_v / total_v * 100) if total_v > 0 else 0
            pct_color = (0, 255, 0) if passed_pct > 50 else (0, 200, 255) if passed_pct > 20 else (0, 0, 255)
            cv2.putText(raw_frame, f"Passed: {passed_v}/{total_v} ({passed_pct:.0f}%)", (mx, 101),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pct_color, 1)

            if video_writer is None:
                h, w = raw_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = AsyncVideoWriter(video_out_path, fourcc, fps, (w, h))
                print(f"Recording: {video_out_path} (async, {fps:.0f} fps)")
            video_writer.write(raw_frame)

            cv2.imshow(name, raw_frame)

            elapsed = time.time() - frame_start_time
            wait_ms = max(1, int((target_frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    # Cleanup
    source.release()
    cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {video_out_path}")

    async_yolo.stop()
    async_ocr.stop()

    print("\n")
    yolo_stats = async_yolo.get_stats()
    print(f"YOLO: submitted={yolo_stats['submitted']}, processed={yolo_stats['processed']}, dropped={yolo_stats['dropped']}")
    async_ocr.print_stats()
    recognizer.finalize()
    speed_tracker.finalize()
    metrics.finalize()

    # Passed rate: recognized plates / unique vehicles
    total_vehicles = speed_tracker.stats["unique_vehicles"]
    passed_plates = len(recognizer.passed_results)
    passed_pct = (passed_plates / total_vehicles * 100) if total_vehicles > 0 else 0
    print(f"\nPassed: {passed_plates}/{total_vehicles} ({passed_pct:.1f}%)")

    print(f"Done: {name}")


if __name__ == "__main__":
    from cli import main
    main()
