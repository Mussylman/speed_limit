# pipeline_builder.py
# Shared pipeline assembly: YOLO, speed estimator, OCR
# Used by both main.py (CLI) and pipeline_processor.py (API/MJPEG)

import os

from config import BASE_DIR


def apply_quality_preset(cfg: dict, quality: str):
    """Mutate cfg dict to apply quality preset settings.

    quality='max' removes all realtime constraints for offline processing.
    quality='default' leaves cfg unchanged.
    """
    if quality != "max":
        return

    cfg["_quality_mode"] = "max"

    # YOLO: larger input, lower confidence threshold
    cfg["yolo_imgsz"] = 1280
    cfg["_yolo_max_queue"] = 64
    cfg["_yolo_min_conf"] = 0.25

    # OCR: disable early-stop, bigger queue, no cooldown
    cfg["_ocr_good_conf"] = 1.0
    cfg["_ocr_max_queue"] = 256
    cfg["ocr_cooldown_frames"] = 0

    # Quality filters: disabled
    cfg["_min_blur_score"] = 0.0
    cfg["_min_brightness"] = 0.0
    cfg["_max_brightness"] = 999.0
    cfg["_quality_improvement"] = 0.0

    # OCR: require 2 confirmations (fuzzy Levenshtein ≤ 1)
    cfg["_min_confirmations"] = 2

    # No frame drops: wait instead of dropping
    cfg["_no_drop"] = True
    cfg["_prefetch_size"] = 64

    # OCR: full resolution crops (no resize)
    cfg["_max_crop_width"] = 0  # 0 = no resize

    # Speed: wider smoothing window
    hom = cfg.get("homography", {})
    hom["smoothing_window"] = 20
    cfg["homography"] = hom


def create_yolo(config: dict):
    """Create YOLO model + AsyncYOLO wrapper.

    Returns (async_yolo, use_half, yolo_imgsz).
    """
    import torch
    from ultralytics import YOLO
    from async_yolo import AsyncYOLO

    yolo_path = os.path.join(BASE_DIR, config["models"]["yolo_model"])
    model = YOLO(yolo_path, task="detect")

    device = config.get("device", "cuda")
    try:
        model.to(device)
    except Exception:
        device = "cpu"

    use_half = device == "cuda" and torch.cuda.is_available()
    yolo_imgsz = int(config.get("yolo_imgsz", 1280))

    async_yolo = AsyncYOLO(
        model=model,
        imgsz=yolo_imgsz,
        classes=[2],  # cars
        half=use_half,
        tracker="bytetrack.yaml",
        max_queue_size=int(config.get("_yolo_max_queue", 3)),
        min_conf=float(config.get("_yolo_min_conf", 0.5)),
    )

    return async_yolo, use_half, yolo_imgsz


def _get_camera_settings(cam_config: dict, camera_id: str) -> dict:
    """Get per-camera settings from config_cam.yaml by camera name."""
    for cam in cam_config.get("cameras", []):
        if cam.get("name") == camera_id:
            return cam
    return {}


def create_speed_estimator(config: dict, cam_config: dict, camera_id: str,
                           fps: float, source=None):
    """Create speed estimator based on config.

    Returns (speed_estimator, speed_method).

    Per-camera settings (homography_config, speed_correction) from
    config_cam.yaml override global config.yaml values.
    """
    from speed_line import SpeedEstimator
    from speed_homography import HomographySpeedEstimator

    cam_settings = _get_camera_settings(cam_config, camera_id)
    speed_method = config.get("speed_method", "lines")

    if speed_method == "homography":
        hom_cfg = config.get("homography", {})

        # Per-camera homography file overrides global
        if cam_settings.get("homography_config"):
            hom_file = os.path.join(BASE_DIR, cam_settings["homography_config"])
        else:
            hom_file = os.path.join(BASE_DIR, hom_cfg.get("config_file", "config/homography_config.yaml"))

        # Per-camera speed_correction overrides global
        speed_correction = cam_settings.get("speed_correction", hom_cfg.get("speed_correction", 1.0))

        if os.path.exists(hom_file):
            speed = HomographySpeedEstimator(
                homography_config=hom_file,
                fps=fps,
                min_track_points=hom_cfg.get("min_track_points", 5),
                smoothing_window=hom_cfg.get("smoothing_window", 10),
                max_speed_kmh=hom_cfg.get("max_speed_kmh", 200),
                min_speed_kmh=hom_cfg.get("min_speed_kmh", 5),
                speed_correction=speed_correction,
            )
            return speed, "homography"
        else:
            # Fallback to lines if homography file not found
            speed_method = "lines"

    # Lines method
    speed = SpeedEstimator(fps)

    # Load line zones
    if source is not None:
        # CLI path: use source info to match camera
        from video.source import SourceType
        line_zones = []
        if source.source_type == SourceType.RTSP and source.info:
            for cam in cam_config.get("cameras", []):
                if cam.get("name") == source.info.name:
                    line_zones = cam.get("line_zones", [])
                    break
        elif "line_zones" in config:
            line_zones = config["line_zones"]
    else:
        # API path: match by camera_id
        line_zones = config.get("line_zones", [])
        for cam in cam_config.get("cameras", []):
            if cam.get("name") == camera_id:
                cam_zones = cam.get("line_zones", [])
                if cam_zones:
                    line_zones = cam_zones
                break

    for zone in line_zones:
        speed.add_line_zone(
            zone["start_line"],
            zone["end_line"],
            zone.get("distance_m", 10),
            direction=zone.get("direction", "down"),
            name=zone.get("name", "Zone"),
            color=tuple(zone.get("color", [0, 255, 0])),
        )

    return speed, "lines"


def create_ocr(config: dict, camera_id: str, output_dir: str, cam_config: dict = None):
    """Create PlateRecognizer + AsyncOCR.

    Per-camera settings (min_car_height, etc.) from config_cam.yaml
    override global config.yaml values.

    Returns (recognizer, async_ocr).
    """
    from plate_recognizer import PlateRecognizer
    from async_ocr import AsyncOCR

    cam_settings = _get_camera_settings(cam_config or {}, camera_id)

    # Per-camera overrides for OCR filters
    def _get(key, default):
        return cam_settings.get(key, config.get(key, default))

    ocr_conf = float(config.get("ocr_conf_threshold", 0.5))
    recognizer = PlateRecognizer(
        output_dir=output_dir,
        camera_id=camera_id,
        min_conf=ocr_conf,
        min_plate_chars=int(config.get("min_plate_chars", 8)),
        min_car_height=int(_get("min_car_height", 150)),
        min_car_width=int(_get("min_car_width", 100)),
        min_plate_width=int(config.get("min_plate_width", 60)),
        min_plate_height=int(config.get("min_plate_height", 15)),
        cooldown_frames=int(config.get("ocr_cooldown_frames", 3)),
        plate_format_regex=config.get("plate_format_regex", ""),
        min_blur_score=float(config.get("_min_blur_score", 5.0)),
        min_brightness=float(config.get("_min_brightness", 20.0)),
        max_brightness=float(config.get("_max_brightness", 240.0)),
        quality_improvement=float(config.get("_quality_improvement", 1.0)),
        min_confirmations=int(config.get("_min_confirmations", 2)),
    )

    max_crop_w = int(config.get("_max_crop_width", 640))

    async_ocr = AsyncOCR(
        recognizer,
        max_queue_size=int(config.get("_ocr_max_queue", 64)),
        num_workers=1,
        max_crop_width=max_crop_w,
        good_conf_threshold=float(config.get("_ocr_good_conf", 0.88)),
        cache_max_size=200,
        cache_ttl_frames=300,
    )

    return recognizer, async_ocr
