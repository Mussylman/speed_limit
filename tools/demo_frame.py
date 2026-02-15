"""Demo: захват кадра 1920x1080 с камеры, overlay метрик, запись 30с видео для WhatsApp."""
import sys, os, time, datetime, subprocess, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from video.decoder import HWDecoder

# === Настройки ===
RTSP_URL = "rtsp://admin:Qaz445566@192.168.18.59:554/stream1"
TARGET_W, TARGET_H = 1920, 1080
DURATION = 30  # секунд записи
BITRATE = 6000  # kbps (WhatsApp лимит ~16MB/мин, 6Mbps ≈ 22MB/30s)

OUT_DIR = r"D:\speed_limit_records"
os.makedirs(OUT_DIR, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(OUT_DIR, f"demo_{ts}.mp4")
snap_path = os.path.join(OUT_DIR, f"demo_{ts}.jpg")

print("=" * 60)
print("  DEMO: camera frame with metrics overlay")
print(f"  Camera:   {RTSP_URL.split('@')[1]}")
print(f"  Target:   {TARGET_W}x{TARGET_H}")
print(f"  Duration: {DURATION}s")
print("=" * 60)

# --- Open camera ---
decoder = HWDecoder(prefer_hw=True)
info = decoder.open(RTSP_URL)
if info.cap is None:
    sys.exit("FAILED: cannot open camera")

src_w, src_h, fps = info.width, info.height, info.fps or 25.0
print(f"Source: {src_w}x{src_h} @ {fps:.1f} fps ({info.backend.value})")

# --- FFmpeg writer (WhatsApp compatible H.264) ---
ffmpeg_bin = shutil.which("ffmpeg")
if not ffmpeg_bin:
    sys.exit("FAILED: ffmpeg not found")

proc = None
for enc, enc_label in [("h264_nvenc", "NVENC"), ("libx264", "x264")]:
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{TARGET_W}x{TARGET_H}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", enc,
        "-b:v", f"{BITRATE}k",
        "-maxrate", f"{BITRATE * 2}k",
        "-bufsize", f"{BITRATE * 2}k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-profile:v", "baseline",  # макс. совместимость с телефонами
        out_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        if proc.poll() is not None:
            proc = None
            continue
        print(f"Encoder: {enc_label}")
        break
    except Exception:
        proc = None

if proc is None:
    sys.exit("FAILED: no encoder")

# --- Capture + overlay ---
frames = 0
errors = 0
start = time.time()
fps_cnt = 0
fps_start = time.time()
cur_fps = 0.0
snapshot_saved = False

print(f"Recording {DURATION}s → {out_path}\n")

while time.time() - start < DURATION:
    ok, frame = decoder.read()
    if not ok or frame is None:
        errors += 1
        continue

    frames += 1
    fps_cnt += 1

    # FPS counter
    now = time.time()
    if now - fps_start >= 1.0:
        cur_fps = fps_cnt / (now - fps_start)
        fps_cnt = 0
        fps_start = now

    # Resize to target
    h, w = frame.shape[:2]
    if w != TARGET_W or h != TARGET_H:
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # === OVERLAY METRICS ===
    elapsed = now - start
    overlay = frame.copy()

    # Semi-transparent header bar
    cv2.rectangle(overlay, (0, 0), (TARGET_W, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Title
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Speed Limit System | {now_str}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Metrics line
    cam_ip = RTSP_URL.split("@")[1].split("/")[0]
    cv2.putText(frame, f"Camera: {cam_ip}  |  {src_w}x{src_h} -> {TARGET_W}x{TARGET_H}  |  "
                        f"FPS: {cur_fps:.1f}  |  Decoder: {info.backend.value}  |  "
                        f"Frame: {frames}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Bottom bar: timer
    bar_y = TARGET_H - 50
    cv2.rectangle(frame, (0, bar_y), (TARGET_W, TARGET_H), (0, 0, 0), -1)
    progress = min(1.0, elapsed / DURATION)
    bar_w = int((TARGET_W - 40) * progress)
    cv2.rectangle(frame, (20, bar_y + 15), (20 + bar_w, bar_y + 35), (0, 200, 0), -1)
    cv2.rectangle(frame, (20, bar_y + 15), (TARGET_W - 20, bar_y + 35), (100, 100, 100), 1)
    cv2.putText(frame, f"{elapsed:.0f}s / {DURATION}s  ({errors} errors)",
                (20, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Write frame
    proc.stdin.write(frame.tobytes())

    # Save first good frame as snapshot
    if not snapshot_saved and frames >= 5:
        cv2.imwrite(snap_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        snapshot_saved = True

    # Show window
    preview = cv2.resize(frame, (960, 540))
    cv2.imshow("Demo", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
decoder.close()
cv2.destroyAllWindows()
proc.stdin.close()
proc.wait(timeout=10)

elapsed = time.time() - start
sz_mb = os.path.getsize(out_path) / (1024 * 1024) if os.path.exists(out_path) else 0

print()
print("=" * 60)
print(f"  Frames:   {frames} ({cur_fps:.1f} fps)")
print(f"  Errors:   {errors}")
print(f"  Duration: {elapsed:.1f}s")
print(f"  File:     {sz_mb:.1f} MB")
print(f"  Video:    {out_path}")
print(f"  Snapshot: {snap_path}")
print("=" * 60)

if sz_mb > 16:
    print(f"  WARNING: file {sz_mb:.1f}MB > 16MB WhatsApp limit!")
    print(f"  Reduce BITRATE or DURATION")
else:
    print(f"  OK: fits WhatsApp limit ({sz_mb:.1f}/16 MB)")
