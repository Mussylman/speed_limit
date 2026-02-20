"""
Diagnostic: model.track() vs model() on Camera_21 test video.
Tests whether ByteTrack is suppressing detections of the moving car.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import torch
from ultralytics import YOLO

VIDEO = os.path.join(os.path.dirname(__file__), '..', 'videos', 'test_12.mp4')
MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo11n.pt')

model = YOLO(MODEL, task="detect")
model.to("cuda")
use_half = True

cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps} fps")

# Test 1: Single frame model.track() vs model() at 70%
target_frame = int(total_frames * 0.7)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()
if not ret:
    sys.exit("Failed to read frame")
print(f"\n=== Test 1: Single frame {target_frame} ===")
print(f"Frame shape: {frame.shape}")

# model() - predict only
res_predict = model(frame, imgsz=1280, conf=0.1, classes=[2], half=use_half, verbose=False)
boxes_p = res_predict[0].boxes
print(f"\nmodel() predict: {len(boxes_p)} detections")
for i, (xyxy, conf) in enumerate(zip(boxes_p.xyxy.cpu().numpy(), boxes_p.conf.cpu().numpy())):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    print(f"  [{i}] box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {w:.0f}x{h:.0f} conf={conf:.3f}")

# model.track() - single call (fresh tracker)
# Need a fresh model to avoid state from predict()
model2 = YOLO(MODEL, task="detect")
model2.to("cuda")
res_track = model2.track(frame, imgsz=1280, conf=0.1, classes=[2], half=use_half,
                         persist=True, tracker="bytetrack.yaml", verbose=False)
boxes_t = res_track[0].boxes
has_ids = boxes_t.id is not None
print(f"\nmodel.track() single frame: {len(boxes_t)} detections, has_ids={has_ids}")
if has_ids:
    for i, (xyxy, conf, tid) in enumerate(zip(
            boxes_t.xyxy.cpu().numpy(), boxes_t.conf.cpu().numpy(),
            boxes_t.id.int().cpu().numpy())):
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        print(f"  [{i}] id={tid} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {w:.0f}x{h:.0f} conf={conf:.3f}")
else:
    for i, (xyxy, conf) in enumerate(zip(boxes_t.xyxy.cpu().numpy(), boxes_t.conf.cpu().numpy())):
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        print(f"  [{i}] NO_ID box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {w:.0f}x{h:.0f} conf={conf:.3f}")

# Test 2: model.track() with accumulated state (simulate pipeline)
print(f"\n=== Test 2: Sequential model.track() (every 6th frame) ===")
model3 = YOLO(MODEL, task="detect")
model3.to("cuda")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0
found_car = False
for i in range(0, total_frames, 6):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, f = cap.read()
    if not ret:
        break

    res = model3.track(f, imgsz=1280, classes=[2], half=use_half,
                       persist=True, tracker="bytetrack.yaml", verbose=False)

    boxes = res[0].boxes
    if boxes.id is None:
        continue

    for j in range(len(boxes.id)):
        xyxy = boxes.xyxy[j].cpu().numpy()
        y2 = xyxy[3]
        conf = float(boxes.conf[j])
        tid = int(boxes.id[j])
        if y2 > 500:  # moving car area (not parked cars at y < 307)
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            print(f"  FOUND: frame={i} id={tid} box=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}] "
                  f"{w:.0f}x{h:.0f} conf={conf:.3f}")
            found_car = True

    if i % 300 == 0:
        n = len(boxes.id) if boxes.id is not None else 0
        ids = [int(x) for x in boxes.id.cpu().numpy()] if boxes.id is not None else []
        print(f"  frame={i}: {n} detections, ids={ids}")

if not found_car:
    print("  NO detections with y2 > 500 in entire video!")

cap.release()
print("\nDone.")
