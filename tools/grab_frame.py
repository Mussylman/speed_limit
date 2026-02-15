"""Сохраняет кадр с RTSP камеры для калибровки"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import argparse
from config import load_configs
from video.source import make_rtsp_url

parser = argparse.ArgumentParser()
parser.add_argument("--camera", required=True)
parser.add_argument("--output", default=None)
args = parser.parse_args()

_, cam_cfg = load_configs()
cred = cam_cfg["camera_credentials"]
cameras = cam_cfg["cameras"]
cam = next((c for c in cameras if c["name"] == args.camera), None)
if not cam:
    sys.exit(f"Camera {args.camera} not found")

url = make_rtsp_url(cred, cam)
print(f"Connecting to {cam['ip']}...")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;16777216|max_delay;1000000"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

# Skip first few frames (may be corrupted)
for _ in range(10):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret:
    sys.exit("Failed to grab frame")

out = args.output or f"frame_{args.camera}.jpg"
cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved: {out} ({frame.shape[1]}x{frame.shape[0]})")
