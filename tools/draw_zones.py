# draw_zones.py
# Interactive tool to draw crop_zone polygon on camera frame.
# Cars inside this zone will have crops collected for OCR.
# Cars outside are tracked by YOLO but ignored for OCR.
#
# Usage:
#   python tools/draw_zones.py --camera Camera_21 --image config/zone_frame_Camera_21.jpg
#   python tools/draw_zones.py --camera Camera_14 --video records/Camera_14_.../video.mp4
#
# Controls:
#   Left click  — add point
#   Right click — undo last point
#   Enter       — save and exit
#   R           — reset all points
#   Q/Esc       — quit without saving

import cv2
import yaml
import argparse
import os
import sys
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config_cam.yaml")

# Original image coordinates (saved to config)
points_orig = []
drawing_done = False

# Display scaling
display_scale = 1.0
orig_frame = None
zone_label = "crop_zone"
WIN = "Draw Zone"


def to_display(x, y):
    return int(x * display_scale), int(y * display_scale)


def to_orig(x, y):
    return int(x / display_scale), int(y / display_scale)


def draw_overlay():
    """Draw polygon overlay on display-sized frame."""
    vis = cv2.resize(orig_frame, None, fx=display_scale, fy=display_scale,
                     interpolation=cv2.INTER_AREA)
    h, w = vis.shape[:2]

    # Convert points to display coordinates
    disp_pts = [to_display(x, y) for x, y in points_orig]

    # Semi-transparent overlay outside polygon
    if len(disp_pts) >= 3:
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array(disp_pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        # Darken outside
        dark = vis.copy()
        dark[mask == 0] = (dark[mask == 0] * 0.3).astype(np.uint8)
        vis = dark
        # Draw filled polygon with transparency
        overlay = vis.copy()
        cv2.fillPoly(overlay, [poly], (0, 180, 0))
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
        # Draw polygon border
        cv2.polylines(vis, [poly], True, (0, 255, 0), 2)

    # Draw points
    for i, (dx, dy) in enumerate(disp_pts):
        cv2.circle(vis, (dx, dy), 6, (0, 255, 255), -1)
        cv2.circle(vis, (dx, dy), 6, (0, 0, 0), 1)
        cv2.putText(vis, str(i + 1), (dx + 10, dy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw lines between consecutive points
    if len(disp_pts) >= 2:
        for i in range(len(disp_pts) - 1):
            cv2.line(vis, disp_pts[i], disp_pts[i + 1], (0, 255, 0), 2)

    # Instructions
    oh, ow = orig_frame.shape[:2]
    cv2.putText(vis, f"{zone_label} ({len(points_orig)} pts) [{ow}x{oh}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, "LClick=add  RClick=undo  R=reset  Enter=save  Q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis


def click_event(event, x, y, flags, param):
    global points_orig, drawing_done

    if drawing_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        ox, oy = to_orig(x, y)
        points_orig.append((ox, oy))
        print(f"  Point {len(points_orig)}: ({ox}, {oy})")
        cv2.imshow(WIN, draw_overlay())

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points_orig:
            removed = points_orig.pop()
            print(f"  Undo: ({removed[0]}, {removed[1]})")
            cv2.imshow(WIN, draw_overlay())


def load_frame(source_path):
    """Load frame from image or video file."""
    ext = os.path.splitext(source_path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        frame = cv2.imread(source_path)
        if frame is None:
            sys.exit(f"Cannot read image: {source_path}")
        return frame

    # Video: grab frame from ~40% (more representative)
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {source_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 100:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.4))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Cannot read frame from: {source_path}")
    return frame


def save_zone(camera_name, zone_points, zone_type="crop_zone"):
    """Save zone polygon to config_cam.yaml under the camera entry."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cameras = config.get("cameras", [])
    found = False
    for cam in cameras:
        if cam.get("name") == camera_name:
            cam[zone_type] = [list(p) for p in zone_points]
            found = True
            break

    if not found:
        sys.exit(f"Camera '{camera_name}' not found in {CONFIG_PATH}")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True, default_flow_style=None)

    print(f"\nSaved {zone_type} ({len(zone_points)} points) for {camera_name}")
    print(f"  Points: {zone_points}")
    print(f"  -> {CONFIG_PATH}")


def main():
    global points_orig, drawing_done, display_scale, orig_frame, zone_label

    parser = argparse.ArgumentParser(
        description="Draw crop_zone polygon on camera frame")
    parser.add_argument("--camera", required=True, help="Camera name (e.g. Camera_21)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to frame image")
    group.add_argument("--video", help="Path to video file (uses frame from ~40%%)")
    parser.add_argument("--zone", default="crop_zone",
                        choices=["crop_zone", "detection_zone"],
                        help="Which zone to draw (default: crop_zone)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Display window width (default: 1280)")
    args = parser.parse_args()

    zone_label = args.zone
    source = args.image or args.video
    orig_frame = load_frame(source)
    h, w = orig_frame.shape[:2]

    # Scale to fit screen
    display_scale = min(args.width / w, 900 / h)
    if display_scale > 1.0:
        display_scale = 1.0
    dw, dh = int(w * display_scale), int(h * display_scale)

    print(f"\nCamera: {args.camera}")
    print(f"Frame: {w}x{h} -> display: {dw}x{dh} (scale: {display_scale:.2f})")
    print(f"Zone: {zone_label}")
    print(f"Coordinates saved in original {w}x{h} space.")
    print(f"\nDraw polygon by clicking points. Close polygon with Enter.\n")

    # Load existing zone if any
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for cam in config.get("cameras", []):
            if cam.get("name") == args.camera and zone_label in cam:
                existing = cam[zone_label]
                print(f"Existing {zone_label}: {existing}")
                print("Drawing new zone will overwrite.\n")
    except Exception:
        pass

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WIN, draw_overlay())
    cv2.setMouseCallback(WIN, click_event)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # Enter
            if len(points_orig) < 3:
                print("Need at least 3 points for a polygon.")
                continue
            drawing_done = True
            save_zone(args.camera, points_orig, zone_label)
            break

        elif key == ord('r') or key == ord('R'):
            points_orig = []
            print("Reset.")
            cv2.imshow(WIN, draw_overlay())

        elif key == ord('q') or key == ord('Q') or key == 27:  # Esc
            print("Cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
