# cli.py
# CLI argument parsing + entry point for speed measurement pipeline

import sys
import argparse

from config import load_configs, print_gpu_info
from video.source import VideoSource, SourceType, create_source_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Speed + ANPR (NomeroffNet)")
    parser.add_argument("--source", required=True, choices=["rtsp", "video", "folder", "image"])
    parser.add_argument("--camera", type=str)
    parser.add_argument("--path", type=str)
    return parser.parse_args()


def create_source_from_args(args, cam_cfg):
    """Build a VideoSource from CLI arguments."""
    if args.source == "rtsp":
        if not args.camera:
            sys.exit("Specify --camera for RTSP source")
        return create_source_from_config(cam_cfg, args.camera, prefer_hw=True)
    else:
        if not args.path:
            sys.exit("Specify --path for this source type")
        source_type = {
            "video": SourceType.VIDEO,
            "folder": SourceType.FOLDER,
            "image": SourceType.IMAGE,
        }[args.source]
        return VideoSource(args.path, source_type=source_type, prefer_hw=True)


def main():
    args = parse_args()
    cfg, cam_cfg = load_configs()
    print_gpu_info()

    source = create_source_from_args(args, cam_cfg)

    from main import process_source
    process_source(source, cfg, cam_cfg)


if __name__ == "__main__":
    main()
