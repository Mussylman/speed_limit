# calibrate_homography.py
# Kalibrovka gomografii (perspektiva -> bird's eye view)
# Klikaete 4 tochki na doroge, vvodite razmery -> poluchaete matricu

import cv2
import numpy as np
import yaml
import argparse
import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

points = []
frame_display = None


def click_event(event, x, y, flags, param):
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"  Tochka {len(points)}: ({x}, {y})")

        cv2.circle(frame_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_display, str(len(points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if len(points) > 1:
            cv2.line(frame_display, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(frame_display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
            print("\n  4 tochki vybrany! Nazhmite lyubuyu klavishu...")

        cv2.imshow("Calibration", frame_display)


def main():
    global frame_display, points

    parser = argparse.ArgumentParser(description="Calibration homography")
    parser.add_argument("--video", required=True, help="Path to video or image")
    parser.add_argument("--camera", type=str, default=None, help="Camera name")
    parser.add_argument("--output", default=None, help="Output file")
    args = parser.parse_args()

    if args.output is None:
        if args.camera:
            args.output = f"config/homography_{args.camera}.yaml"
        else:
            args.output = "homography_config.yaml"

    # Load frame
    if args.video.lower().endswith(('.jpg', '.png', '.jpeg')):
        frame = cv2.imread(args.video)
    else:
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("ERROR: cannot read video")
            return

    if frame is None:
        print(f"ERROR: cannot read {args.video}")
        return

    frame_display = frame.copy()
    h, w = frame.shape[:2]

    print("=" * 60)
    print("  KALIBROVKA GOMOGRAFII")
    print("=" * 60)
    print(f"  Resolution: {w}x{h}")
    if args.camera:
        print(f"  Camera: {args.camera}")
    print(f"  Output: {args.output}")
    print()
    print("  Kliknite 4 tochki na doroge (pryamougolnik):")
    print()
    print("   1 *-----------* 2    (daleko)")
    print("     |           |")
    print("     |  doroga   |")
    print("     |           |")
    print("   4 *-----------* 3    (blizko)")
    print()
    print("  Ispolzuite razmetku ili kraya polosy")
    print("=" * 60)

    cv2.imshow("Calibration", frame_display)
    cv2.setMouseCallback("Calibration", click_event)

    while len(points) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            print("Cancelled")
            cv2.destroyAllWindows()
            return

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Real dimensions
    print()
    print("  Vvedite razmery pryamougolnika:")

    try:
        width_m = float(input("   Shirina (1-2 i 4-3) v metrah [3.5]: ") or "3.5")
        length_m = float(input("   Dlina (1-4 i 2-3) v metrah [20]: ") or "20")
    except ValueError:
        print("ERROR: wrong input")
        return

    # Source points (pixels)
    src_pts = np.float32(points)

    # Destination points (meters, bird's eye view)
    scale = 100  # 100 px = 1 meter
    dst_pts = np.float32([
        [0, 0],
        [width_m * scale, 0],
        [width_m * scale, length_m * scale],
        [0, length_m * scale]
    ])

    # Homography matrix
    H, status = cv2.findHomography(src_pts, dst_pts)

    # Save config
    config = {
        "homography": {
            "src_points": [list(map(int, p)) for p in points],
            "real_width_m": width_m,
            "real_length_m": length_m,
            "scale_px_per_m": scale,
            "matrix": H.tolist(),
            "image_size": [w, h]
        }
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True, default_flow_style=None)

    # Save matrix separately
    matrix_file = args.output.replace(".yaml", "_matrix.npy")
    np.save(matrix_file, H)

    print()
    print("=" * 60)
    print("  KALIBROVKA ZAVERSHENA")
    print("=" * 60)
    print(f"  Config: {args.output}")
    print(f"  Matrix: {matrix_file}")
    print()
    print("  Source points (pixels):")
    for i, p in enumerate(points, 1):
        print(f"   {i}: {p}")
    print()
    print(f"  Real size: {width_m}m x {length_m}m")
    print()
    print("  Homography matrix:")
    print(H)
    print()

    # Show bird's eye view
    print("  Bird's Eye View...")

    bev_w = int(width_m * scale) + 100
    bev_h = int(length_m * scale) + 100

    dst_pts_shifted = dst_pts + np.array([50, 50])
    H_display, _ = cv2.findHomography(src_pts, dst_pts_shifted)

    bev = cv2.warpPerspective(frame, H_display, (bev_w, bev_h))

    # Grid (every meter)
    for i in range(int(width_m) + 1):
        x = 50 + i * scale
        cv2.line(bev, (x, 50), (x, 50 + int(length_m * scale)), (0, 255, 0), 1)
    for i in range(int(length_m) + 1):
        y = 50 + i * scale
        cv2.line(bev, (50, y), (50 + int(width_m * scale), y), (0, 255, 0), 1)

    cv2.putText(bev, "Bird's Eye View (1 cell = 1m)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Bird's Eye View", bev)
    cv2.imshow("Original", frame)
    print("  Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
