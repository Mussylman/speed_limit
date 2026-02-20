"""
Live grid: 3 cameras with YOLO detection + ByteTrack tracking.
Draws bboxes, track IDs, crop zones. No OCR (lightweight).

Usage:
  python tools/grid_live.py
  python tools/grid_live.py --videos v1.mp4 v2.mp4 v3.mp4
  python tools/grid_live.py --speed 2
"""

import os
import sys
import copy
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import load_configs


def find_latest_videos(records_dir, cameras):
    groups = defaultdict(dict)
    for fname in os.listdir(records_dir):
        if not fname.endswith(".mp4") or not os.path.isfile(os.path.join(records_dir, fname)):
            continue
        for cam in cameras:
            if fname.startswith(cam + "_"):
                ts = fname[len(cam) + 1:].replace(".mp4", "")
                groups[ts][cam] = os.path.join(records_dir, fname)
    complete = {ts: p for ts, p in groups.items() if all(c in p for c in cameras)}
    if not complete:
        return None
    latest_ts = sorted(complete.keys())[-1]
    return complete[latest_ts], latest_ts


class TrackerSwapper:
    """Manages per-camera ByteTrack state for a single YOLO model."""

    def __init__(self, model):
        self.model = model
        self._states = {}
        self._current = None

    def swap(self, camera_id):
        if camera_id == self._current:
            return
        predictor = getattr(self.model, 'predictor', None)
        if predictor is None:
            self._current = camera_id
            return
        # Save current
        if self._current is not None and hasattr(predictor, 'trackers'):
            self._states[self._current] = copy.deepcopy(predictor.trackers)
        # Restore target
        saved = self._states.get(camera_id)
        if saved is not None:
            predictor.trackers = copy.deepcopy(saved)
        elif hasattr(predictor, 'trackers'):
            delattr(predictor, 'trackers')
        self._current = camera_id


def draw_detections(frame, results, scale_back=1.0, track_colors=None):
    """Draw bboxes + track IDs on frame. Returns list of (track_id, bbox)."""
    dets = []
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return dets

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    h, w = frame.shape[:2]

    for i in range(len(ids)):
        x1 = int(xyxy[i][0] * scale_back)
        y1 = int(xyxy[i][1] * scale_back)
        x2 = int(xyxy[i][2] * scale_back)
        y2 = int(xyxy[i][3] * scale_back)
        tid = int(ids[i])
        conf = float(confs[i])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Color per track_id
        if track_colors is not None:
            if tid not in track_colors:
                np.random.seed(tid * 7 + 42)
                track_colors[tid] = tuple(int(c) for c in np.random.randint(80, 255, 3))
            color = track_colors[tid]
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: track_id + conf
        label = f"#{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Bottom center dot (for zone checks)
        cx = (x1 + x2) // 2
        cv2.circle(frame, (cx, y2), 4, color, -1)

        dets.append((tid, (x1, y1, x2, y2)))

    return dets


def draw_zone(frame, points, color=(0, 220, 0), alpha=0.12, label="crop_zone"):
    pts = np.array(points, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    cv2.putText(frame, label, (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def make_grid(frames, cell_w=640, cell_h=360):
    cells = [cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
             for f in frames]
    return np.hstack(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs=3, help="3 video paths")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-detect-size", type=int, default=0,
                        help="Resize frame before YOLO (0=no resize)")
    args = parser.parse_args()

    cfg, cam_cfg = load_configs()
    cameras_cfg = {c["name"]: c for c in cam_cfg["cameras"]}
    cam_names = ["Camera_12", "Camera_14", "Camera_21"]

    # Find videos
    if args.videos:
        video_paths = dict(zip(cam_names, args.videos))
    else:
        records_dir = os.path.join(BASE_DIR, "records")
        result = find_latest_videos(records_dir, cam_names)
        if not result:
            sys.exit("No sync videos found")
        video_paths, ts = result
        print(f"Videos: {ts}")

    # Open captures
    caps = {}
    for name in cam_names:
        cap = cv2.VideoCapture(video_paths[name])
        if not cap.isOpened():
            sys.exit(f"Cannot open {video_paths[name]}")
        caps[name] = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {name}: {w}x{h} @{fps:.0f}fps, {total}f")

    # Load YOLO
    import torch
    from ultralytics import YOLO

    yolo_path = os.path.join(BASE_DIR, cfg["models"]["yolo_model"])
    print(f"\nLoading YOLO: {yolo_path}")
    model = YOLO(yolo_path, task="detect")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    use_half = device == "cuda"
    print(f"  Device: {device}, half={use_half}, imgsz={args.imgsz}")

    swapper = TrackerSwapper(model)

    # Crop zones
    zones = {}
    for name in cam_names:
        cz = cameras_cfg.get(name, {}).get("crop_zone")
        if cz:
            zones[name] = cz

    # Per-camera track colors
    track_colors = {name: {} for name in cam_names}

    fps_video = caps[cam_names[0]].get(cv2.CAP_PROP_FPS) or 25
    delay_base = max(1, int(1000 / fps_video / args.speed))
    frame_idx = 0
    paused = False

    # FPS counter
    fps_start = time.time()
    fps_frames = 0
    display_fps = 0.0

    print(f"\nRunning: {fps_video:.0f}fps x{args.speed}, skip={args.skip}, "
          f"detect_resize={args.max_detect_size or 'off'}")
    print("Keys: SPACE=pause Q=quit +/-=speed D=skip30 A=back30")

    while True:
        if not paused:
            frames = {}
            for name in cam_names:
                ok, frame = caps[name].read()
                if not ok:
                    print(f"\nEnd at frame {frame_idx}")
                    caps[name].release()
                    frames = None
                    break
                frames[name] = frame

            if frames is None:
                break

            frame_idx += 1
            fps_frames += 1

            # FPS
            now = time.time()
            if now - fps_start >= 1.0:
                display_fps = fps_frames / (now - fps_start)
                fps_frames = 0
                fps_start = now

            # Run YOLO on each camera (sequential, with tracker swap)
            run_yolo = (args.skip <= 1 or frame_idx % args.skip == 0)
            det_counts = {}

            for name in cam_names:
                frame = frames[name]
                h, w = frame.shape[:2]

                if run_yolo:
                    # Optional resize for detection
                    detect_frame = frame
                    scale_back = 1.0
                    if args.max_detect_size > 0 and h > args.max_detect_size:
                        scale_back = h / args.max_detect_size
                        new_h = args.max_detect_size
                        new_w = int(w / scale_back)
                        detect_frame = cv2.resize(frame, (new_w, new_h),
                                                  interpolation=cv2.INTER_AREA)

                    swapper.swap(name)
                    results = model.track(
                        detect_frame, imgsz=args.imgsz, classes=[2],
                        persist=True, verbose=False, half=use_half,
                        tracker="bytetrack.yaml",
                    )
                    dets = draw_detections(frame, results, scale_back, track_colors[name])
                    det_counts[name] = len(dets)
                else:
                    det_counts[name] = 0

                # Draw zones
                if name in zones:
                    draw_zone(frame, zones[name])

                # Camera label + stats
                label = cameras_cfg.get(name, {}).get("label", "")
                info = f"{name} ({label})  det:{det_counts[name]}"
                cv2.putText(frame, info, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Compose grid
            grid = make_grid([frames[n] for n in cam_names])

            # Global info bar
            total_dets = sum(det_counts.values())
            bar = f"F:{frame_idx}  FPS:{display_fps:.1f}  det:{total_dets}  speed:x{args.speed:.1f}"
            cv2.putText(grid, bar, (10, grid.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live Grid: YOLO + Tracking (Q=quit)", grid)

        key = cv2.waitKey(delay_base) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('+') or key == ord('='):
            args.speed = min(8.0, args.speed * 1.5)
            delay_base = max(1, int(1000 / fps_video / args.speed))
        elif key == ord('-'):
            args.speed = max(0.1, args.speed / 1.5)
            delay_base = max(1, int(1000 / fps_video / args.speed))
        elif key == ord('d'):
            for name in cam_names:
                for _ in range(29):
                    caps[name].read()
            frame_idx += 29
            paused = False
        elif key == ord('a'):
            target = max(0, frame_idx - 30)
            for name in cam_names:
                caps[name].set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_idx = target
            paused = False

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
