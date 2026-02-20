# main.py
# Pipeline: YOLO tracking + speed + OCR (NomeroffNet)
# OCR every frame, best confidence result is kept

import os
import cv2
import time
import datetime
import numpy as np
from collections import deque

from config import BASE_DIR, HEADLESS
from pipeline_builder import create_yolo, create_speed_estimator, create_ocr
from speed_tracker import SpeedTracker
from metrics_logger import MetricsLogger
from file_logger import FileLogger
from report_generator import ReportGenerator
from video.source import SourceType
from video.writer import AsyncVideoWriter


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


def process_source(source, cfg, cam_cfg, cross_queue=None, start_time=None,
                   shared_pipeline=None, nomeroff_lock=None, shared_yolo=None,
                   deferred_ocr=False, camera_result=None, exp_dir=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = source.info.name
    if exp_dir:
        out_dir = os.path.join(exp_dir, name)
    else:
        out_dir = os.path.join(BASE_DIR, cfg["output_dir"], f"{name}_run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    run_start_time = datetime.datetime.now()
    run_start_epoch = time.time()

    print(f"\nStart: {name}")
    if cfg.get("_quality_mode") == "max":
        print(f"{'='*50}")
        print(f"  QUALITY MODE: MAX (offline, no limits)")
        print(f"  yolo_imgsz={cfg.get('yolo_imgsz')}, min_conf={cfg.get('_yolo_min_conf')}")
        print(f"  ocr_cooldown=0, quality_gate=OFF, blur_filter=OFF, min_confirm={cfg.get('_min_confirmations', 2)}")
        print(f"  no_drop={cfg.get('_no_drop', False)}, prefetch={cfg.get('_prefetch_size', 8)}")
        print(f"{'='*50}")
    print(f"Output: {out_dir}")
    print(f"Backend: {source.info.backend}")
    print(f"Resolution: {source.info.width}x{source.info.height} @ {source.info.fps:.1f} fps")

    # --- Pipeline components via builder ---
    async_yolo, use_half, yolo_imgsz = create_yolo(cfg, shared_yolo=shared_yolo,
                                                    camera_id=name)

    if use_half:
        print("YOLO: FP16 mode enabled")

    fps = source.info.fps if source.info.fps > 0 else cfg.get("fps", 30)

    # Video start time: --start-time (manual) > meta.json > filename > file mtime
    video_start_time = start_time  # from CLI --start-time
    if video_start_time is None and source.source_type == SourceType.VIDEO and os.path.isfile(source.url):
        import re, json as _json
        # Try meta.json in same directory (saved by record_stream.py)
        video_dir = os.path.dirname(os.path.abspath(source.url))
        meta_path = os.path.join(video_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
                video_start_time = datetime.datetime.fromtimestamp(meta["start_epoch"])
                print(f"Video start: {video_start_time.strftime('%d-%m-%Y %H:%M:%S')} (from meta.json)")
            except Exception:
                pass
        # Fallback: parse from filename
        if video_start_time is None:
            basename = os.path.basename(source.url)
            m = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', basename)
            if m:
                try:
                    video_start_time = datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
                except ValueError:
                    pass
        # Fallback: file mtime
        if video_start_time is None:
            try:
                mtime = os.path.getmtime(source.url)
                duration = source.info.total_frames / fps if fps > 0 else 0
                video_start_time = datetime.datetime.fromtimestamp(mtime - duration)
            except Exception:
                pass
    if video_start_time and start_time is None:
        print(f"Video start: {video_start_time.strftime('%d-%m-%Y %H:%M:%S')}")

    speed, speed_method = create_speed_estimator(cfg, cam_cfg, name, fps, source=source)
    show_bird_eye = False
    if speed_method == "homography":
        show_bird_eye = cfg.get("homography", {}).get("show_bird_eye", True)
        print("Speed: HOMOGRAPHY")
    else:
        print("Speed: LINES")

    # Per-camera settings (needed before OCR/collector creation)
    from pipeline_builder import _get_camera_settings
    cam_settings = _get_camera_settings(cam_cfg, name)

    # Aspect ratio filter: front/rear view only (width/height ~ 0.9-1.4)
    min_aspect = float(cam_settings.get("min_aspect_ratio", 0.8))
    max_aspect = float(cam_settings.get("max_aspect_ratio", 1.5))
    min_crop_area = int(cam_settings.get("min_crop_area", 30000))
    min_crop_width = int(cam_settings.get("min_crop_width", 350))
    ocr_resize_width = int(cam_settings.get("ocr_resize_width", 800))
    cross_time_offset = float(cam_settings.get("cross_time_offset", 0.0))
    aspect_skipped = 0
    area_skipped = 0
    width_skipped = 0

    # Zones: crop_zone (where to collect crops for OCR)
    crop_zone = None
    crop_zone_raw = cam_settings.get("crop_zone")
    if crop_zone_raw:
        crop_zone = np.array(crop_zone_raw, dtype=np.int32)
        print(f"Crop zone: {len(crop_zone)} points")
    detection_zone = None
    detection_zone_raw = cam_settings.get("detection_zone")
    if detection_zone_raw:
        detection_zone = np.array(detection_zone_raw, dtype=np.int32)
        print(f"Detection zone: {len(detection_zone)} points")

    # OCR or deferred crop collection
    crop_collector = None
    recognizer = None
    async_ocr = None
    if deferred_ocr:
        from crop_collector import CropCollector
        crop_collector = CropCollector(
            max_per_track=int(cfg.get("max_crops_per_track", 10)),
            save_dir=out_dir,
            min_car_height=int(cam_settings.get("min_car_height",
                                                 cfg.get("min_car_height", 150))),
            min_car_width=int(cam_settings.get("min_car_width",
                                                cfg.get("min_car_width", 100))),
            video_start_time=video_start_time,
            video_fps=fps,
        )
        print(f"OCR: DEFERRED (collecting crops, saving ALL to disk)")
        print(f"   Car filter: >= {crop_collector.min_car_height}x{crop_collector.min_car_width} px")
        print(f"   Crops dir: {out_dir}/crops/")
    else:
        recognizer, async_ocr = create_ocr(cfg, name, out_dir, cam_config=cam_cfg,
                                           shared_pipeline=shared_pipeline,
                                           nomeroff_lock=nomeroff_lock)
        if video_start_time is not None:
            recognizer.video_start_time = video_start_time
            recognizer.video_fps = fps
        recognizer.camera_label = cam_settings.get("label", "")
        plate_format = cfg.get("plate_format_regex", "")
        print(f"OCR: NomeroffNet (ASYNC)")
        print(f"   Car: >= {recognizer.min_car_height}x{recognizer.min_car_width} px")
        print(f"   Plate: >= {recognizer.min_plate_width}x{recognizer.min_plate_height} px")
        print(f"   Format: {plate_format if plate_format else 'any'}")
        print(f"   Cooldown: {cfg.get('ocr_cooldown_frames', 3)} frames")
        print(f"   Min confirmations: {recognizer.min_confirmations}")
        print(f"   Aspect ratio: {min_aspect:.1f} - {max_aspect:.1f}, min width: {min_crop_width}, min area: {min_crop_area}")

    # Parameters (per-camera overrides global)
    ocr_conf_threshold = float(cfg.get("ocr_conf_threshold", 0.5))
    frame_skip = int(cam_settings.get("frame_skip", cfg.get("frame_skip", 1)))
    show_window = bool(cfg.get("show_window", True)) and not HEADLESS
    speed_limit = int(cam_settings.get("speed_limit", cfg.get("speed_limit", 70)))
    no_drop = cfg.get("_no_drop", False)

    detect_sz = cfg.get("_yolo_max_detect_size", 0)
    detect_info = f", detect_resize={detect_sz}p" if detect_sz else ""
    print(f"YOLO: imgsz={yolo_imgsz}, skip={frame_skip}{detect_info}")

    # Motion detection: skip YOLO on static frames
    from motion_detector import MotionDetector
    motion = MotionDetector(threshold=5.0, min_area_pct=0.5, warmup_frames=5)

    # Speed tracker (domain: per-vehicle max speed, violations, raw stream)
    speed_tracker = SpeedTracker(
        output_dir=out_dir,
        camera_id=name,
        speed_limit=speed_limit,
        video_start_time=video_start_time,
        video_fps=fps,
    )

    # Metrics
    metrics = MetricsLogger(
        output_dir=out_dir,
        report_interval=10.0,
        json_log=True,
    )
    metrics.async_ocr = async_ocr  # None in deferred mode

    # File logger
    file_logger = FileLogger(output_dir=out_dir)
    if recognizer:
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
    bev_cache = None  # bird's eye view cache

    # Bbox extrapolation: per-track velocity (dx, dy per frame)
    track_prev_pos = {}   # {obj_id: (cx, cy, frame_idx)}
    track_velocity = {}   # {obj_id: (dx_per_frame, dy_per_frame)}

    # Display FPS control
    target_frame_time = 1.0 / fps
    last_display_time = time.time()

    # Display FPS counter
    display_fps = 0.0
    display_fps_frames = 0
    display_fps_start = time.time()

    # Latency tracking
    latency_history = deque(maxlen=300)
    latency_ms = 0.0
    latency_p95 = 0.0
    latency_calc_time = time.time()

    # Cross-camera: track which track_ids already emitted
    cross_emitted = set()

    # Compact status log
    status_interval = 10.0
    last_status_time = time.time()
    prev_dropped = 0
    null_frames = 0
    zone_skipped = 0

    try:
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
            null_frames += 1
            # Video file EOF: prefetch thread ended, queue drained
            if source.source_type == SourceType.VIDEO and source._prefetch_thread is not None:
                if not source._prefetch_thread.is_alive() and source._prefetch_queue.empty():
                    break
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

        # === COMPACT STATUS LOG (every N seconds) ===
        now_status = time.time()
        if now_status - last_status_time >= status_interval:
            dropped_now = source.frames_dropped if hasattr(source, 'frames_dropped') else 0
            new_drops = dropped_now - prev_dropped
            prev_dropped = dropped_now
            dec_fps_val = source.decode_fps if hasattr(source, 'decode_fps') else 0
            q_sz_val = source.queue_size if hasattr(source, 'queue_size') else 0
            q_cap_val = source.queue_capacity if hasattr(source, 'queue_capacity') else 0
            rc = getattr(source, '_reconnect_count', 0)

            yolo_stats = async_yolo.get_stats()

            total_speeds = len(speed_tracker.speeds)
            violations = speed_tracker.stats.get("violations", 0)
            passed_plates = len(recognizer.passed_results) if recognizer else 0

            print(f"\n{'='*70}")
            print(f"[STREAM] FPS:{display_fps:.1f} decode:{dec_fps_val:.0f} queue:{q_sz_val}/{q_cap_val} "
                  f"drop:{new_drops} null:{null_frames} reconnect:{rc} lat:{latency_ms:.0f}ms(p95:{latency_p95:.0f})")
            yolo_sub = yolo_stats['submitted']
            yolo_drop = yolo_stats['dropped']
            yolo_drop_pct = (yolo_drop / yolo_sub * 100) if yolo_sub > 0 else 0
            yolo_lock_avg = yolo_stats.get('gpu_lock_avg_ms', 0)
            yolo_lock_info = f" lock_wait:{yolo_lock_avg:.0f}ms" if yolo_lock_avg > 0 else ""
            print(f"[YOLO]   submit:{yolo_sub} done:{yolo_stats['processed']} "
                  f"drop:{yolo_drop}({yolo_drop_pct:.0f}%) time:{last_yolo_time_ms:.0f}ms{yolo_lock_info}")

            if recognizer:
                ocr_stats = recognizer.stats
                ocr_async = async_ocr.get_stats()
                ocr_lock_cnt = ocr_stats.get('nomeroff_lock_count', 0)
                ocr_lock_avg = (ocr_stats['nomeroff_lock_wait_ms'] / ocr_lock_cnt) if ocr_lock_cnt > 0 else 0
                ocr_lock_info = f" lock_wait:{ocr_lock_avg:.0f}ms" if ocr_lock_avg > 0 else ""
                print(f"[OCR]    called:{ocr_stats['ocr_called']} no_plate:{ocr_stats['ocr_no_plate']}{ocr_lock_info} "
                      f"skip: size={ocr_stats['skipped_car_size']} blur={ocr_stats['skipped_quality']} "
                      f"dup={ocr_stats['skipped_not_better']} cool={ocr_stats['skipped_cooldown']} "
                      f"fmt={ocr_stats['skipped_format']} chars={ocr_stats['skipped_chars']} "
                      f"plate_sz={ocr_stats['skipped_plate_size']} "
                      f"low_conf={ocr_stats['skipped_low_conf']} unconf={ocr_stats['unconfirmed']}")
                print(f"[OCR-Q]  submit:{ocr_async['submitted']} done:{ocr_async['processed']} "
                      f"upgrades:{ocr_async['upgrades']} "
                      f"drop: full={ocr_async['dropped_full']} flight={ocr_async['dropped_in_flight']} "
                      f"max_upg={ocr_async['dropped_max_upgrades']} not_bigger={ocr_async['dropped_not_bigger']}")
            if crop_collector:
                print(f"[CROPS]  {crop_collector.stats_str()} zone_skip:{zone_skipped}")
            print(f"[FILTER] aspect_skip:{aspect_skipped} width_skip:{width_skipped} area_skip:{area_skipped} (ratio:{min_aspect:.1f}-{max_aspect:.1f} w>={min_crop_width} area>={min_crop_area})")
            print(f"[MOTION] {motion.get_stats_str()}")
            print(f"[RESULT] cars:{total_speeds} plates:{passed_plates} violations:{violations}")
            print(f"{'='*70}")

            # Periodic sync: frequency-voted text -> passed_results
            if async_ocr and recognizer:
                async_ocr.sync_to_recognizer(recognizer)

            null_frames = 0
            last_status_time = now_status

        # === SUBMIT TO YOLO (async, every Nth frame, skip static) ===
        submitted_yolo = False
        if frame_skip <= 1 or frame_idx % frame_skip == 0:
            if motion.has_motion(frame):
                async_yolo.submit(frame, frame_idx, blocking=no_drop)
                submitted_yolo = True

        # === GET YOLO RESULTS ===
        # no_drop (quality max): ждём результат синхронно → bbox точно на кадре
        # realtime: забираем что есть без ожидания
        for yolo_result in async_yolo.get_results(blocking=no_drop and submitted_yolo):
            metrics.start_frame(yolo_result.frame_idx)
            last_yolo_time_ms = yolo_result.processing_time_ms
            last_yolo_frame = yolo_result.frame_idx

            # Store full frame (JPEG) for OCR results with OSD timestamp
            _, jpg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if recognizer:
                recognizer.full_frames[yolo_result.frame_idx] = jpg_buf

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
                cx = (x1i + x2i) // 2
                cy = (y1i + y2i) // 2
                bot_center = (cx, y2i)  # bottom-center for zone checks

                # Detection zone: skip objects entirely outside
                if detection_zone is not None:
                    if cv2.pointPolygonTest(detection_zone, (float(bot_center[0]), float(bot_center[1])), False) < 0:
                        continue

                # Считаем скорость bbox (пикселей/кадр) для экстраполяции
                prev = track_prev_pos.get(obj_id)
                if prev:
                    px, py, pf = prev
                    df = yolo_result.frame_idx - pf
                    if df > 0:
                        track_velocity[obj_id] = ((cx - px) / df, (cy - py) / df)
                track_prev_pos[obj_id] = (cx, cy, yolo_result.frame_idx)

                speed.process(yolo_result.frame_idx, obj_id, det.cx, det.cy)

                # Speed (computed first — needed for crop collector)
                current_speed = None
                if speed_method == "homography":
                    v = speed.get_speed(obj_id)
                    if v is not None:
                        current_speed = v
                else:
                    if hasattr(speed, 'object_speeds') and obj_id in speed.object_speeds:
                        v, _ = speed.object_speeds[obj_id]
                        current_speed = v

                # Crop zone check (for realtime OCR mode)
                in_crop_zone = True
                if crop_zone is not None:
                    check_pts = [
                        (float(cx), float(y2i)),
                        (float(x1i), float(y2i)),
                        (float(x2i), float(y2i)),
                        (float(cx), float(cy)),
                    ]
                    in_crop_zone = any(
                        cv2.pointPolygonTest(crop_zone, pt, False) >= 0
                        for pt in check_pts
                    )
                    if not in_crop_zone:
                        zone_skipped += 1

                # OCR or crop collection
                plate_text = ""
                plate_conf = 0.0
                if crop_collector:
                    # Deferred mode: save ALL crops (disk), zone doesn't filter
                    crop_collector.submit(
                        track_id=obj_id,
                        crop=det.crop,
                        full_frame_jpg=jpg_buf,
                        frame_idx=yolo_result.frame_idx,
                        bbox=det.box,
                        speed_kmh=current_speed,
                        detection_conf=det.conf,
                        car_width=x2i - x1i,
                        car_height=y2i - y1i,
                    )
                elif in_crop_zone:
                    # Pre-OCR filters: aspect ratio + crop size
                    car_w = x2i - x1i
                    car_h = y2i - y1i
                    aspect = car_w / max(1, car_h)
                    crop_area = car_w * car_h
                    if aspect < min_aspect or aspect > max_aspect:
                        aspect_skipped += 1
                    elif car_w < min_crop_width:
                        width_skipped += 1
                    elif crop_area < min_crop_area:
                        area_skipped += 1
                    else:
                        # Один качественный resize до 800px (предел NomeroffNet)
                        ocr_crop = det.crop
                        ch, cw = ocr_crop.shape[:2]
                        # Debug: сохраняем ОРИГИНАЛЬНЫЙ кроп (до resize)
                        if recognizer and hasattr(recognizer, 'debug_car_crops'):
                            recognizer._debug_idx += 1
                            cv2.imwrite(os.path.join(recognizer.debug_car_crops,
                                f"{recognizer._debug_idx:05d}_f{yolo_result.frame_idx}_t{obj_id}_orig_{cw}x{ch}.jpg"),
                                det.crop)
                            recognizer._dbg_orig_crop = det.crop  # для plate_bbox на оригинале
                        if ocr_resize_width > 0 and cw < ocr_resize_width:
                            s = ocr_resize_width / cw
                            ocr_crop = cv2.resize(ocr_crop, None, fx=s, fy=s,
                                                  interpolation=cv2.INTER_LANCZOS4)
                        async_ocr.submit(
                            track_id=obj_id,
                            crop=ocr_crop,
                            car_height=car_h,
                            car_width=car_w,
                            frame_idx=yolo_result.frame_idx,
                            detection_conf=det.conf,
                            orig_crop=det.crop,
                        )
                    cached_result = async_ocr.get_cached_result(obj_id)
                    plate_text = cached_result.plate_text if cached_result else ""
                    plate_conf = cached_result.ocr_conf if cached_result else 0.0

                if current_speed is not None:
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
                ocr_queue=async_ocr.queue_size() if async_ocr else 0,
                yolo_queue=async_yolo.input_queue.qsize(),
                cars_count=len(detections),
            )

            if recognizer:
                metrics.update_skip_reasons(recognizer.stats)
            metrics.end_frame()

        # === GET OCR RESULTS ===
        for ocr_result in (async_ocr.get_results(current_frame_idx=frame_idx) if async_ocr else []):
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
                # Cross-camera: emit first confirmed plate per track
                if cross_queue and ocr_result.track_id not in cross_emitted:
                    try:
                        cross_queue.put_nowait({
                            "camera_id": name,
                            "plate_text": ocr_result.result.plate_text,
                            "timestamp": time.time() + cross_time_offset,
                            "frame_idx": ocr_result.frame_idx,
                        })
                        cross_emitted.add(ocr_result.track_id)
                    except Exception:
                        pass  # queue full — skip

        # === DRAWING (every frame) ===
        raw_frame = frame.copy()

        if speed_method == "lines" and hasattr(speed, 'draw_zones'):
            speed.draw_zones(raw_frame)

        # Сколько кадров прошло с последнего YOLO результата
        frame_delta = frame_idx - last_yolo_frame

        for det in cached_detections:
            x1i, y1i, x2i, y2i = det['box']
            obj_id = det['obj_id']
            plate_text = det['plate_text']
            plate_conf = det['plate_conf']
            current_speed = det['speed']

            # Экстраполяция bbox: сдвигаем по скорости трека
            vel = track_velocity.get(obj_id)
            if vel and frame_delta > 0:
                dx = int(vel[0] * frame_delta)
                dy = int(vel[1] * frame_delta)
                h_frame, w_frame = raw_frame.shape[:2]
                x1i = max(0, min(w_frame - 1, x1i + dx))
                y1i = max(0, min(h_frame - 1, y1i + dy))
                x2i = max(0, min(w_frame, x2i + dx))
                y2i = max(0, min(h_frame, y2i + dy))

            # Пропускаем мелкие детекции (тени, далёкие объекты)
            bw, bh = x2i - x1i, y2i - y1i
            if bw < 80 or bh < 60:
                continue

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

        # --- Overlays for recording (always) ---
        if recognizer and recognizer.passed_results:
            raw_frame = draw_results_panel(
                raw_frame, recognizer.passed_results,
                speed_tracker, speed_limit
            )

        if show_bird_eye and speed_method == "homography":
            if last_yolo_frame == frame_idx or bev_cache is None:
                bev_cache = speed.draw_bird_eye_view(size=(300, 400), frame_idx=frame_idx)
            bev = bev_cache
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
        passed_v = len(recognizer.passed_results) if recognizer else 0
        passed_pct = (passed_v / total_v * 100) if total_v > 0 else 0
        pct_color = (0, 255, 0) if passed_pct > 50 else (0, 200, 255) if passed_pct > 20 else (0, 0, 255)
        cv2.putText(raw_frame, f"Passed: {passed_v}/{total_v} ({passed_pct:.0f}%)", (mx, 101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pct_color, 1)

        # --- Video recording (always) ---
        if video_writer is None:
            h, w = raw_frame.shape[:2]
            video_writer = AsyncVideoWriter(video_out_path, fps, (w, h))
            print(f"Recording: {video_out_path} (async, {fps:.0f} fps, {w}x{h})")
        video_writer.write(raw_frame)

        # --- Window display (optional) ---
        if show_window:
            cv2.imshow(name, raw_frame)

            elapsed = time.time() - frame_start_time
            wait_ms = max(1, int((target_frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
        else:
            # Headless: allow Ctrl+C, no FPS throttle
            pass
    except KeyboardInterrupt:
        print("\nCtrl+C — saving results...")

    # Cleanup
    source.release()
    if show_window:
        cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {video_out_path}")

    async_yolo.stop()

    print("\n")
    yolo_stats = async_yolo.get_stats()
    ys, yd = yolo_stats['submitted'], yolo_stats['dropped']
    yd_pct = (yd / ys * 100) if ys > 0 else 0
    print(f"YOLO: submitted={ys}, processed={yolo_stats['processed']}, dropped={yd} ({yd_pct:.1f}%)")

    if deferred_ocr:
        # Deferred mode: return data for post-processing (OCR runs after all cameras)
        print(f"Crops: {crop_collector.stats_str()} zone_skip:{zone_skipped}")
        metrics.finalize()
        if camera_result is not None:
            camera_result.update({
                'collector': crop_collector,
                'speed_tracker': speed_tracker,
                'out_dir': out_dir,
                'run_start_time': run_start_time,
                'duration_sec': time.time() - run_start_epoch,
                'video_out_path': video_out_path if video_writer is not None else None,
            })
        print(f"Done (live): {name}")
    else:
        async_ocr.stop()
        async_ocr.print_stats()
        # Sync frequency-voted winners back to plate_recognizer before saving
        async_ocr.sync_to_recognizer(recognizer, final=True)
        recognizer.finalize()
        speed_tracker.finalize()
        metrics.finalize()

        # --- Client report ---
        duration_sec = time.time() - run_start_epoch
        coords = cfg.get("coordinates")
        coords = tuple(coords) if coords else None
        report_gen = ReportGenerator(out_dir, name, speed_limit,
                                         address=cfg.get("address", ""),
                                         coordinates=coords)
        vid_path = video_out_path if video_writer is not None else None
        report_gen.generate(
            passed_results=recognizer.passed_results,
            speed_tracker=speed_tracker,
            video_path=vid_path,
            start_time=run_start_time,
            duration_sec=duration_sec,
        )

        # Passed rate: recognized plates / unique vehicles
        total_vehicles = speed_tracker.stats["unique_vehicles"]
        passed_plates = len(recognizer.passed_results)
        passed_pct = (passed_plates / total_vehicles * 100) if total_vehicles > 0 else 0
        print(f"\nPassed: {passed_plates}/{total_vehicles} ({passed_pct:.1f}%)")

        print(f"Done: {name}")


if __name__ == "__main__":
    from cli import main
    main()
