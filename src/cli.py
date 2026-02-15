# cli.py
# CLI argument parsing + entry point for speed measurement pipeline

import sys
import argparse
import multiprocessing

from config import load_configs, print_gpu_info
from pipeline_builder import apply_quality_preset
from video.source import VideoSource, SourceType, create_source_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Speed + ANPR (NomeroffNet)")
    parser.add_argument("--source", required=True, choices=["rtsp", "video", "folder", "image"])
    parser.add_argument("--camera", type=str, nargs="+", help="Camera name(s) from config_cam.yaml")
    parser.add_argument("--path", type=str)
    parser.add_argument("--quality", type=str, default="default", choices=["default", "max"],
                        help="Quality preset: default (live) or max (offline, no limits)")
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


def _run_camera(camera_name, cfg, cam_cfg):
    """Entry point for a single camera process."""
    from main import process_source

    print(f"[{camera_name}] Starting...")
    no_drop = cfg.get("_no_drop", False)
    prefetch_size = cfg.get("_prefetch_size", 8)
    source = create_source_from_config(
        cam_cfg, camera_name, prefer_hw=True,
        no_drop=no_drop, prefetch_size=prefetch_size,
    )
    process_source(source, cfg, cam_cfg)


def main():
    args = parse_args()
    cfg, cam_cfg = load_configs()
    apply_quality_preset(cfg, args.quality)
    print_gpu_info()

    if args.source == "rtsp" and args.camera and len(args.camera) > 1:
        # Multi-camera: each camera in a separate process
        print(f"Starting {len(args.camera)} cameras: {', '.join(args.camera)}")
        processes = []
        for cam_name in args.camera:
            p = multiprocessing.Process(
                target=_run_camera,
                args=(cam_name, cfg, cam_cfg),
                name=cam_name,
            )
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nStopping all cameras...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=5)
    else:
        # Single camera / video / folder / image
        source = create_source_from_args(args, cfg, cam_cfg)
        from main import process_source
        process_source(source, cfg, cam_cfg)


if __name__ == "__main__":
    main()
