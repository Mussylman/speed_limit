"""
Diagnostic: Simulate SharedAsyncYOLO's tracker swap with 3 cameras.
Compares: single-camera vs interleaved with tracker swap.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from ultralytics import YOLO

BASE = os.path.join(os.path.dirname(__file__), '..')
MODEL = os.path.join(BASE, 'models', 'yolo11n.pt')
VIDEOS = {
    'Camera_12': os.path.join(BASE, 'videos', 'test_10.mp4'),
    'Camera_14': os.path.join(BASE, 'videos', 'test_11.mp4'),
    'Camera_21': os.path.join(BASE, 'videos', 'test_12.mp4'),
}

model = YOLO(MODEL, task="detect")
model.to("cuda")

# Open all videos
caps = {}
for cam, path in VIDEOS.items():
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{cam}: {path} ({total} frames)")
    caps[cam] = cap

FRAME_SKIP = 6
IMGSZ = 1280

# Track state per camera (simulate SharedAsyncYOLO)
camera_trackers = {cam: None for cam in VIDEOS}
last_camera = None

def swap_tracker(camera_id):
    global last_camera
    if camera_id == last_camera:
        return

    predictor = getattr(model, 'predictor', None)
    if predictor is None:
        last_camera = camera_id
        return

    # Save current
    if last_camera is not None and hasattr(predictor, 'trackers'):
        camera_trackers[last_camera] = copy.deepcopy(predictor.trackers)

    # Restore target
    saved = camera_trackers.get(camera_id)
    if saved is not None:
        predictor.trackers = copy.deepcopy(saved)
    else:
        if hasattr(predictor, 'trackers'):
            delattr(predictor, 'trackers')

    last_camera = camera_id


print(f"\n=== Interleaved 3-camera tracking (frame_skip={FRAME_SKIP}) ===")

# Process frames interleaved like SharedAsyncYOLO would
cam_order = ['Camera_12', 'Camera_14', 'Camera_21']
cam21_detections_y500 = []

max_frames = 1750
for fi in range(0, max_frames, FRAME_SKIP):
    for cam in cam_order:
        cap = caps[cam]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        swap_tracker(cam)

        res = model.track(
            frame, imgsz=IMGSZ, classes=[2], half=True,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )

        boxes = res[0].boxes
        if boxes.id is None:
            continue

        if cam == 'Camera_21':
            for j in range(len(boxes.id)):
                xyxy = boxes.xyxy[j].cpu().numpy()
                y2 = xyxy[3]
                conf = float(boxes.conf[j])
                tid = int(boxes.id[j])
                if y2 > 500:
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    cam21_detections_y500.append((fi, tid, conf, w, h, y2))
                    print(f"  Cam21 FOUND: frame={fi} id={tid} "
                          f"box=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}] "
                          f"{w:.0f}x{h:.0f} conf={conf:.3f}")

    if fi % 300 == 0:
        print(f"  Progress: frame {fi}/{max_frames}")

if cam21_detections_y500:
    print(f"\nCamera_21: {len(cam21_detections_y500)} detections with y2>500")
else:
    print(f"\nCamera_21: ZERO detections with y2>500 (BUG REPRODUCED)")

for cap in caps.values():
    cap.release()
print("Done.")
