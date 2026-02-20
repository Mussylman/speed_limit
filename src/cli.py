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


def _next_exp_dir(output_base):
    """Find next exp_N directory number and create it."""
    os.makedirs(output_base, exist_ok=True)
    existing = []
    for d in os.listdir(output_base):
        if os.path.isdir(os.path.join(output_base, d)) and d.startswith("exp_"):
            try:
                existing.append(int(d.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass
    next_num = max(existing, default=0) + 1
    exp_dir = os.path.join(output_base, f"exp_{next_num}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Speed + ANPR (NomeroffNet)")
    parser.add_argument("--source", required=True, choices=["rtsp", "video", "folder", "image"])
    parser.add_argument("--camera", type=str, nargs="+", help="Camera name(s) from config_cam.yaml")
    parser.add_argument("--path", type=str)
    parser.add_argument("--quality", type=str, default="default", choices=["default", "max"],
                        help="Quality preset: default (live) or max (offline, no limits)")
    parser.add_argument("--start-time", type=str,
                        help="Video start time from camera OSD, e.g. '16-02-2026 13:31:41'")
    parser.add_argument("--no-defer", action="store_true",
                        help="Multi-camera: realtime OCR instead of deferred post-processing")
    parser.add_argument("--name", type=str, default="",
                        help="Test name for cross-camera report (e.g. '15 km/h')")
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
                shared_pipeline=None, nomeroff_lock=None, shared_yolo=None,
                video_path=None, deferred_ocr=False, camera_results=None,
                exp_dir=None):
    """Entry point for a single camera thread (RTSP or video file)."""
    from main import process_source

    print(f"[{camera_name}] Starting...")
    no_drop = cfg.get("_no_drop", False)
    prefetch_size = cfg.get("_prefetch_size", 8)
    if video_path:
        source = VideoSource(
            video_path, source_type=SourceType.VIDEO, prefer_hw=True,
            no_drop=no_drop, prefetch_size=prefetch_size,
        )
        if source.info:
            source.info.name = camera_name
    else:
        source = create_source_from_config(
            cam_cfg, camera_name, prefer_hw=True,
            no_drop=no_drop, prefetch_size=prefetch_size,
        )
    result_data = {}
    process_source(source, cfg, cam_cfg, cross_queue=cross_queue,
                   shared_pipeline=shared_pipeline,
                   nomeroff_lock=nomeroff_lock,
                   shared_yolo=shared_yolo,
                   deferred_ocr=deferred_ocr,
                   camera_result=result_data,
                   exp_dir=exp_dir)
    if camera_results is not None and result_data:
        camera_results[camera_name] = result_data


def _generate_cross_report(cam_dirs, exp_dir, cam_cfg, name=""):
    """Generate cross-camera speed report from sequential results."""
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "tools"))
        from cross_camera_report import load_config, collect_results, match_plates, draw_report

        distances, speed_limit = load_config()
        events = collect_results(cam_dirs)
        if not events:
            return
        results = match_plates(events, distances)
        if not results:
            print("\n[CROSS-CAM] No cross-camera matches found.")
            return

        import json
        # Save JSON
        json_path = os.path.join(exp_dir, "cross_camera_speeds.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"name": name, "speed_limit": speed_limit,
                        "results": results}, f, indent=2, ensure_ascii=False)

        # Draw image
        report_name = name or os.path.basename(exp_dir)
        img_path = os.path.join(exp_dir, "cross_camera_report.jpg")
        draw_report(results, report_name, speed_limit, img_path)

        # Print summary
        print(f"\n{'='*60}")
        print(f"[CROSS-CAM] Cross-Camera Speed Report")
        print(f"{'='*60}")
        for r in results:
            marker = " !!!" if r["speed_kmh"] > speed_limit else ""
            print(f"  {r['plate']:12s} {r['cam_from']}->{r['cam_to']:12s} "
                  f"{r['distance_m']:6.1f}m / {r['time_sec']:5.1f}s = "
                  f"{r['speed_kmh']:6.1f} km/h{marker}")
        print(f"{'='*60}")
        print(f"Report: {img_path}")
    except Exception as e:
        print(f"\n[CROSS-CAM] Error generating report: {e}")


def main():
    args = parse_args()
    cfg, cam_cfg = load_configs()
    apply_quality_preset(cfg, args.quality)
    print_gpu_info()

    # Multi-camera: RTSP with multiple --camera, or video with --paths
    video_paths = {}  # camera_name -> video_path
    multi_camera = False

    if args.source == "rtsp" and args.camera and len(args.camera) > 1:
        multi_camera = True
    elif args.source == "video" and args.path and args.camera and len(args.camera) > 1:
        import glob as _glob
        paths = [p.strip() for p in args.path.split(";")]
        if len(paths) == 1 and os.path.isdir(paths[0]):
            # Single directory: auto-find video for each camera
            base_dir = os.path.abspath(paths[0])
            for cam_name in args.camera:
                # Try Camera_XX_*.mp4 pattern
                matches = sorted(_glob.glob(os.path.join(base_dir, f"{cam_name}_*.mp4")))
                if matches:
                    video_paths[cam_name] = matches[-1]  # latest
                else:
                    print(f"Warning: no video for {cam_name} in {base_dir}")
        elif len(paths) == len(args.camera):
            for cam_name, p in zip(args.camera, paths):
                p = os.path.abspath(p)
                if os.path.isdir(p):
                    fixed = _glob.glob(os.path.join(p, "*_fixed.mp4"))
                    if fixed:
                        video_paths[cam_name] = fixed[0]
                    else:
                        mp4s = _glob.glob(os.path.join(p, "*.mp4"))
                        if mp4s:
                            video_paths[cam_name] = mp4s[0]
                        else:
                            sys.exit(f"No .mp4 files in {p}")
                else:
                    video_paths[cam_name] = p
        else:
            sys.exit(f"Number of paths ({len(paths)}) must match cameras ({len(args.camera)})")
        multi_camera = True

    # Experiment directory for this run
    output_base = os.path.join(BASE_DIR, cfg.get("output_dir", "outputs"))
    exp_dir = _next_exp_dir(output_base)
    print(f"\nExperiment: {os.path.basename(exp_dir)}")

    if multi_camera and video_paths:
        # Sequential video processing — one camera at a time, one exp_dir
        print(f"Sequential video processing: {', '.join(args.camera)}")
        cfg["show_window"] = False

        cam_dirs = []
        for cam_name in args.camera:
            vpath = video_paths.get(cam_name)
            if not vpath:
                print(f"[{cam_name}] No video found, skipping")
                continue
            print(f"\n{'='*60}")
            print(f"Start: {cam_name}")
            print(f"{'='*60}")
            _run_camera(cam_name, cfg, cam_cfg, video_path=vpath, exp_dir=exp_dir)
            cam_dir = os.path.join(exp_dir, cam_name)
            if os.path.isdir(cam_dir):
                cam_dirs.append(cam_dir)

        # Auto cross-camera report
        if len(cam_dirs) >= 2:
            _generate_cross_report(cam_dirs, exp_dir, cam_cfg, args.name)

    elif multi_camera:
        # Threaded RTSP mode
        mode = "RTSP"
        print(f"Starting {len(args.camera)} cameras (threaded, shared GPU, {mode}): "
              f"{', '.join(args.camera)}")

        # Auto-optimize for multi-camera
        cfg["show_window"] = False
        use_deferred = not args.no_defer

        # Shared GPU resources
        from pipeline_builder import create_shared_yolo
        shared_yolo, _, _ = create_shared_yolo(cfg)

        shared_pipeline = None
        nomeroff_lock = None
        if not use_deferred:
            # Direct mode: load NomeroffNet now alongside YOLO
            from pipeline_builder import create_shared_nomeroff_pipeline
            shared_pipeline = create_shared_nomeroff_pipeline(cfg)
            nomeroff_lock = threading.Lock()
            cfg.setdefault("_ocr_max_scales", 2)  # realtime: fast scales
            print("OCR mode: DIRECT (realtime)")
        else:
            cfg.setdefault("_ocr_max_scales", 6)  # post-processing: full scales
            print("OCR mode: DEFERRED (post-processing)")

        camera_results = {}  # filled by camera threads for post-processing

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
                output_dir=exp_dir,
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
                    "shared_yolo": shared_yolo,
                    "shared_pipeline": shared_pipeline,
                    "nomeroff_lock": nomeroff_lock,
                    "video_path": video_paths.get(cam_name),
                    "deferred_ocr": use_deferred,
                    "camera_results": camera_results,
                    "exp_dir": exp_dir,
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
            for t in threads:
                t.join(timeout=10)

        shared_yolo.stop()

        # Post-processing (deferred mode only)
        if use_deferred and camera_results:
            from pipeline_builder import create_shared_nomeroff_pipeline
            from crop_collector import post_process_ocr
            shared_pipeline = create_shared_nomeroff_pipeline(cfg)
            post_process_ocr(camera_results, shared_pipeline, cfg, cam_cfg)

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
        process_source(source, cfg, cam_cfg, start_time=manual_start, exp_dir=exp_dir)


if __name__ == "__main__":
    main()
