# cli.py
# CLI argument parsing + entry point for speed measurement pipeline

import os
import sys
import argparse
import threading
import queue

from config import BASE_DIR, load_configs, print_gpu_info
from pipeline_builder import apply_quality_preset
from video.source import VideoSource, SourceType, create_source_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Speed + ANPR (NomeroffNet)")
    parser.add_argument("--source", required=True, choices=["rtsp", "video", "folder", "image"])
    parser.add_argument("--camera", type=str, nargs="+", help="Camera name(s) from config_cam.yaml")
    parser.add_argument("--path", type=str)
    parser.add_argument("--quality", type=str, default="default", choices=["default", "max"],
                        help="Quality preset: default (live) or max (offline, no limits)")
    parser.add_argument("--start-time", type=str,
                        help="Video start time from camera OSD, e.g. '16-02-2026 13:31:41'")
    return parser.parse_args()


def create_source_from_args(args, cfg, cam_cfg):
    """Build a VideoSource from CLI arguments (single camera)."""
    no_drop = cfg.get("_no_drop", False)
    prefetch_size = cfg.get("_prefetch_size", 8)

    if args.source == "rtsp":
        if not args.camera:
            sys.exit("Specify --camera for RTSP source")
        return create_source_from_config(
            cam_cfg, args.camera[0], prefer_hw=True,
            no_drop=no_drop, prefetch_size=prefetch_size,
        )
    else:
        if not args.path:
            sys.exit("Specify --path for this source type")
        source_type = {
            "video": SourceType.VIDEO,
            "folder": SourceType.FOLDER,
            "image": SourceType.IMAGE,
        }[args.source]
        return VideoSource(
            args.path, source_type=source_type, prefer_hw=True,
            no_drop=no_drop, prefetch_size=prefetch_size,
        )


def _run_camera(camera_name, cfg, cam_cfg, cross_queue=None,
                shared_pipeline=None, nomeroff_lock=None, yolo_lock=None):
    """Entry point for a single camera thread."""
    from main import process_source

    print(f"[{camera_name}] Starting...")
    no_drop = cfg.get("_no_drop", False)
    prefetch_size = cfg.get("_prefetch_size", 8)
    source = create_source_from_config(
        cam_cfg, camera_name, prefer_hw=True,
        no_drop=no_drop, prefetch_size=prefetch_size,
    )
    process_source(source, cfg, cam_cfg, cross_queue=cross_queue,
                   shared_pipeline=shared_pipeline,
                   nomeroff_lock=nomeroff_lock,
                   yolo_lock=yolo_lock)


def main():
    args = parse_args()
    cfg, cam_cfg = load_configs()
    apply_quality_preset(cfg, args.quality)
    print_gpu_info()

    if args.source == "rtsp" and args.camera and len(args.camera) > 1:
        # Multi-camera: threads sharing single GPU context
        print(f"\nStarting {len(args.camera)} cameras (threaded, shared GPU): "
              f"{', '.join(args.camera)}")

        # Auto-optimize for multi-camera live
        cfg["show_window"] = False
        cfg.setdefault("_ocr_max_scales", 2)  # fast: original + 640px only
        cfg.setdefault("_yolo_max_detect_size", 1080)  # resize before YOLO

        # Shared GPU resources: 1x NomeroffNet + locks
        from pipeline_builder import create_shared_nomeroff_pipeline
        shared_pipeline = create_shared_nomeroff_pipeline(cfg)
        nomeroff_lock = threading.Lock()
        yolo_lock = threading.Lock()

        # Cross-camera average speed (queue.Queue instead of mp.Queue)
        cross_cfg = cam_cfg.get("cross_camera", {})
        cross_tracker = None
        cross_queue = None
        if cross_cfg.get("enabled", False) and cross_cfg.get("distances_m"):
            from cross_camera_speed import CrossCameraTracker
            cross_queue = queue.Queue(maxsize=1000)
            cross_tracker = CrossCameraTracker(
                distances_config=cross_cfg["distances_m"],
                speed_limit=int(cross_cfg.get("speed_limit", 70)),
                output_dir=os.path.join(BASE_DIR, cfg.get("output_dir", "output")),
            )
            cross_tracker.start(cross_queue)
            print(f"[CROSS-CAM] Enabled: {len(cross_cfg['distances_m'])} distance pairs, "
                  f"limit={cross_cfg.get('speed_limit', 70)} km/h")

        threads = []
        for cam_name in args.camera:
            t = threading.Thread(
                target=_run_camera,
                args=(cam_name, cfg, cam_cfg, cross_queue),
                kwargs={
                    "shared_pipeline": shared_pipeline,
                    "nomeroff_lock": nomeroff_lock,
                    "yolo_lock": yolo_lock,
                },
                name=cam_name,
            )
            t.start()
            threads.append(t)

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\nStopping all cameras...")
            # Threads will exit via their own KeyboardInterrupt / source.release()
            for t in threads:
                t.join(timeout=10)

        if cross_tracker:
            cross_tracker.finalize()
            cross_tracker.print_summary()
    else:
        # Single camera / video / folder / image
        source = create_source_from_args(args, cfg, cam_cfg)
        # Override source name with --camera so per-camera config is applied
        if args.camera and source.info:
            source.info.name = args.camera[0]
        # Parse --start-time
        import datetime
        manual_start = None
        if args.start_time:
            try:
                manual_start = datetime.datetime.strptime(args.start_time, "%d-%m-%Y %H:%M:%S")
            except ValueError:
                sys.exit(f"Invalid --start-time format. Use: DD-MM-YYYY HH:MM:SS")
        from main import process_source
        process_source(source, cfg, cam_cfg, start_time=manual_start)


if __name__ == "__main__":
    main()
