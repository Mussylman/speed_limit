"""
Grid preview: 3 cameras side-by-side with crop_zone overlay.
Shows where crop_collector would trigger (green polygon = crop zone).

Usage:
  python tools/grid_preview.py
  python tools/grid_preview.py --videos path1.mp4 path2.mp4 path3.mp4
"""

import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cam_config():
    path = os.path.join(BASE_DIR, "config", "config_cam.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_videos(records_dir, cameras):
    """Find most recent set of sync-recorded videos (same timestamp in filename)."""
    # Group by timestamp suffix
    from collections import defaultdict
    groups = defaultdict(dict)
    for fname in os.listdir(records_dir):
        if not fname.endswith(".mp4"):
            continue
        fpath = os.path.join(records_dir, fname)
        if not os.path.isfile(fpath):
            continue
        for cam in cameras:
            if fname.startswith(cam + "_"):
                # Extract timestamp: Camera_12_2026-02-17_14-25-43.mp4
                ts = fname[len(cam) + 1:].replace(".mp4", "")
                groups[ts][cam] = fpath

    # Find latest group that has all cameras
    complete = {ts: paths for ts, paths in groups.items()
                if all(c in paths for c in cameras)}
    if not complete:
        return None
    latest_ts = sorted(complete.keys())[-1]
    return complete[latest_ts], latest_ts


def draw_zone(frame, points, color=(0, 220, 0), alpha=0.15, label="crop_zone"):
    """Draw semi-transparent polygon zone with label."""
    pts = np.array(points, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    # Label at top-left of zone
    x_min = pts[:, 0].min()
    y_min = pts[:, 1].min()
    cv2.putText(frame, label, (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def make_grid(frames, labels, cell_w=640, cell_h=360):
    """Arrange frames in 1x3 horizontal grid."""
    cells = []
    for i, (frame, label) in enumerate(zip(frames, labels)):
        cell = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        # Camera label
        cv2.putText(cell, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cells.append(cell)

    # 1x3 horizontal
    return np.hstack(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs=3, help="3 video paths (cam12, cam14, cam21)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    cam_cfg = load_cam_config()
    cameras_cfg = {c["name"]: c for c in cam_cfg["cameras"]}
    cam_names = ["Camera_12", "Camera_14", "Camera_21"]

    if args.videos:
        video_paths = {name: path for name, path in zip(cam_names, args.videos)}
    else:
        records_dir = os.path.join(BASE_DIR, "records")
        result = find_latest_videos(records_dir, cam_names)
        if not result:
            sys.exit("No sync videos found for all 3 cameras in records/")
        video_paths, ts = result
        print(f"Using latest sync recording: {ts}")

    # Open captures
    caps = {}
    for name in cam_names:
        path = video_paths[name]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"Cannot open {path}")
        caps[name] = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {name}: {w}x{h} @ {fps:.1f} fps, {frames_total} frames — {path}")

    # Get crop zones
    zones = {}
    for name in cam_names:
        cfg = cameras_cfg.get(name, {})
        cz = cfg.get("crop_zone")
        if cz:
            zones[name] = np.array(cz, dtype=np.int32)
            print(f"  {name} crop_zone: {len(cz)} points")

    # Labels
    labels = {}
    for name in cam_names:
        cfg = cameras_cfg.get(name, {})
        labels[name] = f"{name} ({cfg.get('label', '')})"

    # Playback
    fps = caps[cam_names[0]].get(cv2.CAP_PROP_FPS) or 25
    delay_ms = max(1, int(1000 / fps / args.speed))
    frame_idx = 0
    paused = False

    print(f"\nPlayback: {fps:.0f} fps x{args.speed} (delay={delay_ms}ms)")
    print("Controls: SPACE=pause, Q=quit, LEFT/RIGHT=skip, +/-=speed")

    while True:
        if not paused:
            frames = {}
            all_ok = True
            for name in cam_names:
                ok, frame = caps[name].read()
                if not ok:
                    all_ok = False
                    break
                frames[name] = frame

            if not all_ok:
                print(f"End of video at frame {frame_idx}")
                break

            # Draw zones
            for name in cam_names:
                if name in zones:
                    draw_zone(frames[name], zones[name], (0, 220, 0), 0.15, "crop_zone")

                # Frame counter
                cv2.putText(frames[name], f"F:{frame_idx}", (10, frames[name].shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            grid = make_grid(
                [frames[n] for n in cam_names],
                [labels[n] for n in cam_names],
            )
            frame_idx += 1

        cv2.imshow("3-Camera Grid (Q=quit, SPACE=pause)", grid)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "PLAYING")
        elif key == ord('+') or key == ord('='):
            args.speed = min(8.0, args.speed * 1.5)
            delay_ms = max(1, int(1000 / fps / args.speed))
            print(f"Speed: x{args.speed:.1f}")
        elif key == ord('-'):
            args.speed = max(0.1, args.speed / 1.5)
            delay_ms = max(1, int(1000 / fps / args.speed))
            print(f"Speed: x{args.speed:.1f}")
        elif key == 83 or key == ord('d'):  # RIGHT arrow
            # Skip 30 frames
            for name in cam_names:
                for _ in range(29):
                    caps[name].read()
            frame_idx += 29
            paused = False
        elif key == 81 or key == ord('a'):  # LEFT arrow
            # Rewind 30 frames
            target = max(0, frame_idx - 30)
            for name in cam_names:
                caps[name].set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_idx = target
            paused = False

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
