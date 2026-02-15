"""Quick test: RTSP capture + FFmpeg NVENC write. No YOLO, no OCR."""
import sys, os, time, subprocess, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

RTSP_URL = "rtsp://admin:Qaz445566@192.168.18.59:554/stream1"
DURATION = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "records")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 50)
print("TEST: RTSP capture + NVENC write")
print(f"URL: {RTSP_URL}")
print(f"Duration: {DURATION}s")
print("=" * 50)

# --- Decoder ---
from video.decoder import HWDecoder
decoder = HWDecoder(prefer_hw=True)
info = decoder.open(RTSP_URL)
if info.cap is None:
    sys.exit("FAILED: cannot open RTSP")

w, h, fps = info.width, info.height, info.fps or 30.0
print(f"Decoder: {info.backend.value} | {w}x{h} @ {fps} fps")

# --- FFmpeg NVENC writer ---
out_path = os.path.join(OUT_DIR, "test_nvenc.mp4")
ffmpeg_bin = shutil.which("ffmpeg")
proc = None
enc_name = "none"

for enc, name in [("h264_nvenc", "NVENC (GPU)"), ("libx264", "libx264 (CPU)")]:
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", enc,
        "-b:v", "8000k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        if proc.poll() is not None:
            proc = None
            continue
        enc_name = name
        break
    except:
        proc = None

if proc is None:
    sys.exit("FAILED: no encoder available")

print(f"Encoder: {enc_name}")
print(f"Output: {out_path}")
print()

# --- Capture loop ---
frames = 0
errors = 0
start = time.time()
last_report = start
fps_cnt = 0
fps_start = time.time()
cur_fps = 0.0

print("Recording...")
while time.time() - start < DURATION:
    ok, frame = decoder.read()
    if ok and frame is not None:
        frames += 1
        fps_cnt += 1
        proc.stdin.write(frame.tobytes())
    else:
        errors += 1

    now = time.time()
    if now - fps_start >= 1.0:
        cur_fps = fps_cnt / (now - fps_start)
        fps_cnt = 0
        fps_start = now
    if now - last_report >= 5.0:
        print(f"  [{now - start:.0f}s] frames={frames} err={errors} fps={cur_fps:.1f}")
        last_report = now

elapsed = time.time() - start
decoder.close()
proc.stdin.close()
proc.wait(timeout=10)

sz = os.path.getsize(out_path) / (1024*1024) if os.path.exists(out_path) else 0
br = (sz * 8) / elapsed if elapsed > 0 else 0

print()
print("=" * 50)
print(f"  Frames:   {frames} | Errors: {errors}")
print(f"  Duration: {elapsed:.1f}s | FPS: {frames/elapsed:.1f}")
print(f"  File:     {sz:.1f} MB | Bitrate: {br:.1f} Mbit/s")
print(f"  Encoder:  {enc_name}")
print(f"  Decoder:  {info.backend.value}")
print("=" * 50)
if errors == 0:
    print("STATUS: OK")
else:
    print(f"STATUS: {errors} errors ({errors/max(frames,1)*100:.1f}%)")
