# crop_collector.py
# Live phase: collect car crops per track (no OCR)
# Post-processing: run OCR on collected crops after all cameras finish

import os
import re
import cv2
import json
import uuid
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class CropRecord:
    """Single car crop with metadata from live capture."""
    track_id: int
    frame_idx: int
    timestamp: datetime
    crop: np.ndarray
    full_frame_jpg: Optional[np.ndarray]  # cv2.imencode buffer
    bbox: Tuple[int, int, int, int]
    speed_kmh: Optional[float]
    detection_conf: float
    car_width: int
    car_height: int
    blur: float
    brightness: float
    quality: float


class CropCollector:
    """Collects car crops per track during live capture.

    Saves ALL crops to disk for inspection.
    Keeps best K in memory for OCR post-processing.
    """

    def __init__(self, max_per_track=10, save_dir=None,
                 min_car_height=40, min_car_width=30,
                 min_blur=5.0, min_brightness=20.0, max_brightness=240.0,
                 video_start_time=None, video_fps=25.0):
        self.max_per_track = max_per_track
        self.save_dir = save_dir
        self.min_car_height = min_car_height
        self.min_car_width = min_car_width
        self.min_blur = min_blur
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.video_start_time = video_start_time
        self.video_fps = video_fps

        self._tracks: Dict[int, List[CropRecord]] = {}
        self.total_submitted = 0
        self.total_accepted = 0
        self.total_saved_disk = 0
        self.rejected_size = 0
        self.rejected_quality = 0

        # Disk crop index: track_id -> [metadata dicts]
        self._disk_index: Dict[int, List[dict]] = {}

        if self.save_dir:
            os.makedirs(os.path.join(self.save_dir, "crops"), exist_ok=True)

    def submit(self, track_id, crop, full_frame_jpg, frame_idx, bbox,
               speed_kmh, detection_conf, car_width, car_height) -> bool:
        self.total_submitted += 1

        # Quick quality (blur + brightness on downscaled grayscale)
        blur, brightness = self._quick_quality(crop)

        # Quality score
        quality = blur * (car_width * car_height) / 10000
        bright_penalty = 1.0 - abs(brightness - 120) / 120
        quality *= max(0.1, bright_penalty)

        # ALWAYS save to disk (before any filtering)
        if self.save_dir:
            self._save_to_disk(track_id, frame_idx, crop, bbox,
                               car_width, car_height, detection_conf,
                               blur, brightness, quality, speed_kmh)

        # Size filter (only for in-memory OCR list)
        if car_height < self.min_car_height or car_width < self.min_car_width:
            self.rejected_size += 1
            return False

        # Quality filter (only for in-memory OCR list)
        if blur < self.min_blur:
            self.rejected_quality += 1
            return False
        if brightness < self.min_brightness or brightness > self.max_brightness:
            self.rejected_quality += 1
            return False

        # Timestamp from video time
        if self.video_start_time:
            ts = self.video_start_time + timedelta(seconds=frame_idx / self.video_fps)
        else:
            ts = datetime.now()

        record = CropRecord(
            track_id=track_id, frame_idx=frame_idx, timestamp=ts,
            crop=crop.copy(), full_frame_jpg=full_frame_jpg,
            bbox=bbox, speed_kmh=speed_kmh, detection_conf=detection_conf,
            car_width=car_width, car_height=car_height,
            blur=blur, brightness=brightness, quality=quality,
        )

        if track_id not in self._tracks:
            self._tracks[track_id] = []

        records = self._tracks[track_id]
        records.append(record)

        # Keep best K by quality in memory
        if len(records) > self.max_per_track:
            records.sort(key=lambda r: r.quality, reverse=True)
            dropped = records.pop()
            dropped.crop = None
            dropped.full_frame_jpg = None

        self.total_accepted += 1
        return True

    def _save_to_disk(self, track_id, frame_idx, crop, bbox,
                      car_width, car_height, detection_conf,
                      blur, brightness, quality, speed_kmh):
        """Save every crop to disk for visual inspection."""
        track_dir = os.path.join(self.save_dir, "crops", f"track_{track_id:04d}")
        os.makedirs(track_dir, exist_ok=True)

        fname = f"f{frame_idx:04d}_{car_width}x{car_height}_c{detection_conf:.2f}.jpg"
        fpath = os.path.join(track_dir, fname)
        cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.total_saved_disk += 1

        # Store metadata for index
        if track_id not in self._disk_index:
            self._disk_index[track_id] = []
        self._disk_index[track_id].append({
            "file": fname,
            "frame": frame_idx,
            "bbox": list(bbox),
            "car_w": car_width,
            "car_h": car_height,
            "conf": round(detection_conf, 3),
            "blur": round(blur, 1),
            "brightness": round(brightness, 1),
            "quality": round(quality, 1),
            "speed": round(speed_kmh, 1) if speed_kmh else None,
        })

    def save_index(self):
        """Save crops_index.json with metadata for all saved crops."""
        if not self.save_dir or not self._disk_index:
            return
        index = {}
        for tid, crops in self._disk_index.items():
            index[f"track_{tid:04d}"] = {
                "count": len(crops),
                "crops": crops,
            }
        path = os.path.join(self.save_dir, "crops", "index.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"  Crops index: {path} ({len(index)} tracks, {self.total_saved_disk} files)")

    def _quick_quality(self, img):
        h, w = img.shape[:2]
        scale = 64 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        else:
            small = img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        return blur, brightness

    def get_all(self) -> Dict[int, List[CropRecord]]:
        """Return all tracks with crops sorted by quality (best first)."""
        result = {}
        for track_id, records in self._tracks.items():
            records.sort(key=lambda r: r.quality, reverse=True)
            result[track_id] = records
        return result

    @property
    def num_tracks(self):
        return len(self._tracks)

    @property
    def num_crops(self):
        return sum(len(r) for r in self._tracks.values())

    def stats_str(self):
        return (f"tracks:{self.num_tracks} crops:{self.num_crops} "
                f"disk:{self.total_saved_disk} "
                f"submit:{self.total_submitted} accept:{self.total_accepted} "
                f"skip_sz:{self.rejected_size} skip_q:{self.rejected_quality}")


def post_process_ocr(camera_results, shared_pipeline, cfg, cam_cfg):
    """Run OCR on all collected crops after live phase.

    camera_results: Dict[cam_name, {collector, speed_tracker, out_dir, ...}]
    shared_pipeline: NomeroffNet pipeline (loaded after live phase)
    """
    from plate_recognizer import PlateRecognizer, PlateEvent
    from kz_plate import fix_kz_plate, merge_texts_charwise
    from pipeline_builder import _get_camera_settings
    from report_generator import ReportGenerator

    plate_format = cfg.get("plate_format_regex", "")
    total_tracks = sum(r['collector'].num_tracks for r in camera_results.values())
    total_crops = sum(r['collector'].num_crops for r in camera_results.values())

    print(f"\n{'='*60}")
    print(f"  POST-PROCESSING OCR")
    print(f"  {len(camera_results)} cameras, {total_tracks} tracks, {total_crops} crops")
    print(f"{'='*60}")

    for cam_name, result in camera_results.items():
        collector = result['collector']
        speed_tracker = result['speed_tracker']
        out_dir = result['out_dir']

        # Save disk index for this camera
        collector.save_index()

        cam_settings = _get_camera_settings(cam_cfg, cam_name)
        camera_label = cam_settings.get("label", "")

        # Create recognizer for this camera (reuses shared pipeline, no lock needed)
        recognizer = PlateRecognizer(
            output_dir=out_dir,
            camera_id=cam_name,
            min_conf=float(cfg.get("ocr_conf_threshold", 0.5)),
            min_plate_chars=int(cfg.get("min_plate_chars", 8)),
            min_car_height=0,  # already filtered by collector
            min_car_width=0,
            min_plate_width=int(cam_settings.get("min_plate_width",
                                                  cfg.get("min_plate_width", 60))),
            min_plate_height=int(cam_settings.get("min_plate_height",
                                                   cfg.get("min_plate_height", 15))),
            cooldown_frames=0,
            plate_format_regex=plate_format,
            min_blur_score=0,
            min_brightness=0,
            max_brightness=999,
            quality_improvement=0,
            min_confirmations=1,
            shared_pipeline=shared_pipeline,
            nomeroff_lock=None,
            ocr_max_scales=int(cfg.get("_ocr_max_scales", 6)),
        )
        recognizer.camera_label = camera_label
        if collector.video_start_time:
            recognizer.video_start_time = collector.video_start_time
            recognizer.video_fps = collector.video_fps

        all_crops = collector.get_all()
        print(f"\n[{cam_name}] Processing {len(all_crops)} tracks "
              f"({collector.num_crops} crops)...")

        passed_count = 0
        failed_count = 0
        t_start = time.time()

        max_ocr_per_track = int(cfg.get("max_ocr_per_track", 5))

        for i, (track_id, crops) in enumerate(all_crops.items()):
            # Run OCR on best crops
            ocr_results = []
            for crop_rec in crops[:max_ocr_per_track]:
                plate_info = recognizer._detect_plate(crop_rec.crop)
                if plate_info:
                    text, plate_conf, ocr_conf, pw, ph, region = plate_info
                    ocr_results.append(
                        (text, plate_conf, ocr_conf, pw, ph, region, crop_rec))
                    # Early stop: 3 good 8-char results is enough for voting
                    good_8 = sum(1 for t, *_ in ocr_results if len(t) == 8)
                    if good_8 >= 3:
                        break

            if not ocr_results:
                failed_count += 1
                continue

            # Character voting across crops (for 8-char KZ plates)
            kz_results = [(t, cr) for t, pc, oc, pw, ph, reg, cr
                          in ocr_results if len(t) == 8]
            if len(kz_results) >= 2:
                texts = [t for t, _ in kz_results]
                merged = merge_texts_charwise(texts)
                best_text = fix_kz_plate(merged)
            else:
                best = max(ocr_results, key=lambda r: r[1])
                best_text = best[0]

            # Format validation
            if plate_format and not re.match(plate_format, best_text):
                failed_count += 1
                best_crop = crops[0]
                event = _make_event(best_crop, best_text, cam_name, camera_label,
                                    ocr_results[0], plate_format)
                event.reject_reason = f"format:{best_text}"
                recognizer._update_failed(track_id, event)
                continue

            # Passed!
            passed_count += 1
            best_crop = crops[0]
            best_ocr = max(ocr_results, key=lambda r: r[1])
            event = _make_event(best_crop, best_text, cam_name, camera_label,
                                best_ocr, plate_format)

            # Store full frame for card saving
            if best_crop.full_frame_jpg is not None:
                recognizer.full_frames[best_crop.frame_idx] = best_crop.full_frame_jpg

            recognizer._update_passed(track_id, event)

            # Progress
            if (i + 1) % 5 == 0 or i == len(all_crops) - 1:
                elapsed = time.time() - t_start
                print(f"  [{cam_name}] OCR: {i+1}/{len(all_crops)} tracks "
                      f"({elapsed:.1f}s) passed:{passed_count} failed:{failed_count}")

        # Finalize recognizer (saves passed/failed images + JSON)
        recognizer.finalize()
        speed_tracker.finalize()

        # Report
        speed_limit = int(cam_settings.get("speed_limit", cfg.get("speed_limit", 70)))
        duration_sec = result.get('duration_sec', 0)
        coords = cfg.get("coordinates")
        coords = tuple(coords) if coords else None
        report_gen = ReportGenerator(out_dir, cam_name, speed_limit,
                                     address=cfg.get("address", ""),
                                     coordinates=coords)
        report_gen.generate(
            passed_results=recognizer.passed_results,
            speed_tracker=speed_tracker,
            video_path=result.get('video_out_path'),
            start_time=result.get('run_start_time'),
            duration_sec=duration_sec,
        )

        total_vehicles = speed_tracker.stats["unique_vehicles"]
        passed_plates = len(recognizer.passed_results)
        pct = (passed_plates / total_vehicles * 100) if total_vehicles > 0 else 0
        print(f"[{cam_name}] Passed: {passed_plates}/{total_vehicles} ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  POST-PROCESSING COMPLETE")
    print(f"{'='*60}")


def _calculate_scores(blur, brightness, plate_width, plate_height,
                      detection_conf, text_len, format_valid=True):
    """Three separate scores (0.0-1.0): car, plate, ocr. Mirrors PlateRecognizer logic."""
    car_score = max(0.5, min(1.0, detection_conf))

    blur_norm = min(1.0, max(0.3, (blur - 10000) / 15000 * 0.7 + 0.3))
    if 80 <= brightness <= 160:
        bright_norm = 1.0
    elif 60 <= brightness < 80 or 160 < brightness <= 200:
        bright_norm = 0.7
    else:
        bright_norm = 0.4
    size_norm = min(1.0, max(0.5, (plate_width - 60) / 180 + 0.5))
    plate_score = blur_norm * 0.4 + bright_norm * 0.2 + size_norm * 0.4

    if text_len >= 8:
        len_norm = 1.0
    elif text_len >= 6:
        len_norm = 0.7
    elif text_len >= 4:
        len_norm = 0.4
    else:
        len_norm = 0.2
    format_norm = 1.0 if format_valid else 0.6
    ocr_score = len_norm * 0.6 + format_norm * 0.4

    total = (car_score + plate_score + ocr_score) / 3
    return {
        "car_score": round(car_score, 3),
        "plate_score": round(plate_score, 3),
        "ocr_score": round(ocr_score, 3),
        "total": round(total, 3),
    }


def _compute_ocr_confidence(text, nomeroff_ocr_conf, plate_format_regex=""):
    """Real OCR confidence based on text structure + NomeroffNet conf."""
    if not text:
        return 0.0
    score = 0.0
    if len(text) >= 8:
        score += 0.4
    elif len(text) >= 7:
        score += 0.2
    if plate_format_regex:
        if re.match(plate_format_regex, text):
            score += 0.4
    else:
        score += 0.2
    if len(text) >= 8:
        d_ok = all(c.isdigit() for c in text[:3]) and all(c.isdigit() for c in text[6:8])
        l_ok = all(c.isalpha() for c in text[3:6])
        if d_ok and l_ok:
            score += 0.2
    text_conf = min(1.0, score)
    if nomeroff_ocr_conf > 0:
        return nomeroff_ocr_conf * 0.7 + text_conf * 0.3
    return text_conf


def _make_event(crop_rec, text, cam_name, camera_label, ocr_tuple,
                plate_format_regex=""):
    """Create PlateEvent from CropRecord + OCR result tuple."""
    from plate_recognizer import PlateEvent
    text_raw, plate_conf, ocr_conf, pw, ph, region = ocr_tuple[:6]

    format_valid = True
    if plate_format_regex:
        format_valid = bool(re.match(plate_format_regex, text))

    scores = _calculate_scores(
        blur=crop_rec.blur,
        brightness=crop_rec.brightness,
        plate_width=pw,
        plate_height=ph,
        detection_conf=crop_rec.detection_conf,
        text_len=len(text),
        format_valid=format_valid,
    )
    real_ocr_conf = _compute_ocr_confidence(text, ocr_conf, plate_format_regex)

    return PlateEvent(
        event_id=str(uuid.uuid4())[:8],
        timestamp=crop_rec.timestamp.isoformat(),
        camera_id=cam_name,
        camera_label=camera_label,
        track_id=crop_rec.track_id,
        frame_idx=crop_rec.frame_idx,
        car_score=scores["car_score"],
        plate_score=scores["plate_score"],
        ocr_score=scores["ocr_score"],
        total_score=scores["total"],
        detection_conf=crop_rec.detection_conf,
        plate_conf=plate_conf,
        ocr_conf=real_ocr_conf,
        plate_text=text,
        region=region,
        brightness=round(crop_rec.brightness, 1),
        blur=round(crop_rec.blur, 1),
        plate_width_px=pw,
        plate_height_px=ph,
        car_width_px=crop_rec.car_width,
        car_height_px=crop_rec.car_height,
        crop=crop_rec.crop,
    )
