# pipeline_builder.py
# Shared pipeline assembly: YOLO, speed estimator, OCR
# Used by both main.py (CLI) and pipeline_processor.py (API/MJPEG)

import os

from config import BASE_DIR


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
    )

    return async_yolo, use_half, yolo_imgsz


def create_speed_estimator(config: dict, cam_config: dict, camera_id: str,
                           fps: float, source=None):
    """Create speed estimator based on config.

    Returns (speed_estimator, speed_method).

    If source is provided and speed_method == "lines", loads zones from
    source/cam_config (used by main.py CLI path). Otherwise loads zones
    from config + cam_config by camera_id (used by API path).
    """
    from speed_line import SpeedEstimator
    from speed_homography import HomographySpeedEstimator

    speed_method = config.get("speed_method", "lines")

    if speed_method == "homography":
        hom_cfg = config.get("homography", {})
        hom_file = os.path.join(BASE_DIR, hom_cfg.get("config_file", "config/homography_config.yaml"))

        if os.path.exists(hom_file):
            speed = HomographySpeedEstimator(
                homography_config=hom_file,
                fps=fps,
                min_track_points=hom_cfg.get("min_track_points", 5),
                smoothing_window=hom_cfg.get("smoothing_window", 10),
                max_speed_kmh=hom_cfg.get("max_speed_kmh", 200),
                min_speed_kmh=hom_cfg.get("min_speed_kmh", 5),
                speed_correction=hom_cfg.get("speed_correction", 1.0),
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


def create_ocr(config: dict, camera_id: str, output_dir: str):
    """Create PlateRecognizer + AsyncOCR.

    Returns (recognizer, async_ocr).
    """
    from plate_recognizer import PlateRecognizer
    from async_ocr import AsyncOCR

    ocr_conf = float(config.get("ocr_conf_threshold", 0.5))
    recognizer = PlateRecognizer(
        output_dir=output_dir,
        camera_id=camera_id,
        min_conf=ocr_conf,
        min_plate_chars=int(config.get("min_plate_chars", 8)),
        min_car_height=int(config.get("min_car_height", 150)),
        min_car_width=int(config.get("min_car_width", 100)),
        min_plate_width=int(config.get("min_plate_width", 60)),
        min_plate_height=int(config.get("min_plate_height", 15)),
        cooldown_frames=int(config.get("ocr_cooldown_frames", 3)),
        plate_format_regex=config.get("plate_format_regex", ""),
    )

    async_ocr = AsyncOCR(
        recognizer,
        max_queue_size=64,
        num_workers=1,
        max_crop_width=640,
        good_conf_threshold=0.88,
        cache_max_size=200,
        cache_ttl_frames=300,
    )

    return recognizer, async_ocr
