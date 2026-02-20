# tools/record_stream.py
# Record RTSP stream to file using FFmpeg (no re-encoding, -c copy).
#
# Features:
#   - Zero CPU/GPU load (remux only, no decode/encode)
#   - Auto-reconnect on WiFi drops (waits and retries)
#   - Each reconnect creates a new file (no data loss)
#   - Shows file size growth as progress indicator
#
# Usage:
#   python tools/record_stream.py --camera Camera_21
#   python tools/record_stream.py --camera Camera_21 Camera_14
#
# Then process offline:
#   python src/cli.py --source video --path records/Camera_21_.../Camera_21_*.mp4 --quality max

import os
import sys
import time
import argparse
import subprocess
import datetime

sys.stdout.reconfigure(line_buffering=True)  # flush prints immediately

# Add src/ to path for config loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import load_configs, BASE_DIR
from video.source import make_rtsp_url


def _file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def _save_stop_meta(meta_path, meta, stop_time, elapsed, size_mb):
    """Update meta JSON with stop time and duration."""
    import json
    meta["stop_epoch"] = stop_time
    meta["stop_time"] = datetime.datetime.fromtimestamp(stop_time).isoformat()
    meta["duration_sec"] = round(elapsed, 1)
    meta["size_mb"] = round(size_mb, 1)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def record_camera(
    rtsp_url: str,
    camera_name: str,
    output_dir: str,
    max_reconnects: int = 999,
    reconnect_delay: float = 5.0,
):
    """Record RTSP stream using FFmpeg -c copy (no re-encoding).

    Each connection attempt creates a new .mp4 file.
    Auto-reconnects on failure with increasing delay.
    """
    os.makedirs(output_dir, exist_ok=True)
    reconnect_count = 0
    total_files = 0
    total_mb = 0.0

    while reconnect_count < max_reconnects:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(output_dir, f"{camera_name}_{timestamp}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-buffer_size", "33554432",     # 32MB buffer
            "-max_delay", "2000000",        # 2s — wait for delayed packets
            "-timeout", "10000000",         # 10s connection timeout
            "-i", rtsp_url,
            "-c", "copy",
            "-movflags", "+frag_keyframe",  # fragmented mp4: survives crash
            out_path,
        ]

        print(f"[{camera_name}] Connecting... -> {os.path.basename(out_path)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Wait a bit to see if connection succeeds
            time.sleep(3)
            retcode = proc.poll()

            if retcode is not None:
                # FFmpeg exited immediately — connection failed
                stderr_out = proc.stderr.read().decode(errors="replace")
                last_lines = stderr_out.strip().split("\n")[-3:]
                for line in last_lines:
                    if line.strip():
                        print(f"[{camera_name}] {line.strip()}")

                reconnect_count += 1
                delay = min(reconnect_delay * reconnect_count, 30)
                print(f"[{camera_name}] Connection failed. "
                      f"Retry {reconnect_count}/{max_reconnects} in {delay:.0f}s...")
                time.sleep(delay)
                continue

            # Connected! Reset reconnect counter
            reconnect_count = 0
            start_time = time.time()
            last_report = start_time
            print(f"[{camera_name}] Recording started!")

            # Save meta per camera with start time
            import json
            meta_path = os.path.join(output_dir, f"meta_{camera_name}.json")
            meta = {
                "camera": camera_name,
                "file": os.path.basename(out_path),
                "start_epoch": start_time,
                "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            while True:
                retcode = proc.poll()
                if retcode is not None:
                    stop_time = time.time()
                    elapsed = stop_time - start_time
                    size_mb = _file_size_mb(out_path)
                    total_mb += size_mb
                    total_files += 1
                    _save_stop_meta(meta_path, meta, stop_time, elapsed, size_mb)
                    print(f"[{camera_name}] Stream ended after {elapsed:.0f}s "
                          f"({size_mb:.1f} MB). Total: {total_files} files, {total_mb:.1f} MB")
                    break

                # Progress every 30s
                now = time.time()
                if now - last_report >= 30:
                    elapsed = now - start_time
                    size_mb = _file_size_mb(out_path)
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    bitrate = (size_mb * 8) / elapsed if elapsed > 0 else 0
                    print(f"[{camera_name}] {mins:02d}:{secs:02d} | "
                          f"{size_mb:.1f} MB | {bitrate:.1f} Mbit/s")
                    last_report = now

                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n[{camera_name}] Stopping...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            stop_time = time.time()
            elapsed = stop_time - start_time
            size_mb = _file_size_mb(out_path)
            total_mb += size_mb
            total_files += 1
            _save_stop_meta(meta_path, meta, stop_time, elapsed, size_mb)
            print(f"[{camera_name}] Saved: {out_path} ({size_mb:.1f} MB)")
            print(f"[{camera_name}] Total: {total_files} files, {total_mb:.1f} MB")
            return

        except Exception as e:
            print(f"[{camera_name}] Error: {e}")

        # Auto-reconnect
        reconnect_count += 1
        delay = min(reconnect_delay * reconnect_count, 30)
        print(f"[{camera_name}] Reconnecting ({reconnect_count}/{max_reconnects}) "
              f"in {delay:.0f}s...")
        time.sleep(delay)

    print(f"[{camera_name}] Max reconnects ({max_reconnects}) reached.")
    print(f"[{camera_name}] Total: {total_files} files, {total_mb:.1f} MB in {output_dir}")


def remux_fragmented(mp4_path):
    """Remux fragmented MP4 to proper MP4 with moov atom (for OpenCV)."""
    fixed_path = mp4_path.replace(".mp4", "_fixed.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-c", "copy", "-movflags", "+faststart",
        fixed_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(fixed_path):
            size_mb = os.path.getsize(fixed_path) / (1024 * 1024)
            # Replace original with fixed
            os.remove(mp4_path)
            os.rename(fixed_path, mp4_path)
            print(f"  Remuxed: {os.path.basename(mp4_path)} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  Remux failed: {os.path.basename(mp4_path)}")
            return False
    except Exception as e:
        print(f"  Remux error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Record RTSP streams (FFmpeg -c copy, no re-encoding)",
        epilog="Then process offline: python src/cli.py --source video --path <file.mp4> --quality max",
    )
    parser.add_argument("--camera", type=str, nargs="+", required=True,
                        help="Camera name(s) from config_cam.yaml")
    parser.add_argument("--name", type=str, default=None,
                        help="Test name (e.g. '15' for 15 km/h). Saves to records/<name>/")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: records/)")
    args = parser.parse_args()

    _, cam_cfg = load_configs()
    cred = cam_cfg.get("camera_credentials", {})
    cameras = cam_cfg.get("cameras", [])

    out_base = args.output or os.path.join(BASE_DIR, "records")
    if args.name:
        out_base = os.path.join(out_base, args.name)
    session_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if len(args.camera) == 1:
        cam_name = args.camera[0]
        cam = next((c for c in cameras if c["name"] == cam_name), None)
        if not cam:
            sys.exit(f"Camera '{cam_name}' not found in config_cam.yaml")

        url = make_rtsp_url(cred, cam)
        out_dir = out_base if args.name else os.path.join(out_base, f"{cam_name}_{session_ts}")

        print(f"{'='*50}")
        print(f"  RTSP RECORDER")
        print(f"  Camera:  {cam_name} ({cam['ip']})")
        print(f"  Mode:    -c copy (zero CPU, no quality loss)")
        print(f"  Output:  {out_dir}")
        if args.name:
            print(f"  Test:    {args.name}")
        print(f"  Reconnect: auto (unlimited)")
        print(f"{'='*50}")
        print(f"Press Ctrl+C to stop\n")

        record_camera(url, cam_name, out_dir)
    else:
        # Multi-camera: each in a process, all files in same directory
        import multiprocessing
        out_dir = out_base  # all cameras save to same dir
        os.makedirs(out_dir, exist_ok=True)

        print(f"{'='*50}")
        print(f"  RTSP RECORDER — {len(args.camera)} cameras")
        if args.name:
            print(f"  Test:    {args.name}")
        print(f"  Output:  {out_dir}")
        print(f"  Mode:    -c copy (zero CPU)")
        print(f"{'='*50}")

        processes = []
        for cam_name in args.camera:
            cam = next((c for c in cameras if c["name"] == cam_name), None)
            if not cam:
                print(f"WARNING: Camera '{cam_name}' not found, skipping")
                continue

            url = make_rtsp_url(cred, cam)

            p = multiprocessing.Process(
                target=record_camera,
                args=(url, cam_name, out_dir),
                name=cam_name,
            )
            p.start()
            processes.append(p)
            print(f"  {cam_name} ({cam['ip']})")

        print(f"\nPress Ctrl+C to stop\n")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nStopping all recordings...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=5)

        # Auto-remux all recorded files (fragmented MP4 -> proper MP4)
        print(f"\nRemuxing videos...")
        import glob
        for mp4 in sorted(glob.glob(os.path.join(out_dir, "*.mp4"))):
            if "_fixed" not in mp4:
                remux_fragmented(mp4)
        print("Done.")


if __name__ == "__main__":
    main()
