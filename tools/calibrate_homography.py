# calibrate_homography.py
# Kalibrovka gomografii (perspektiva -> bird's eye view)
# Klikaete 4 tochki na doroge, vvodite razmery -> poluchaete matricu
#
# Kadry bolshogo razresheniya (2560x1440+) umenshayutsya dlya otobrazheniya,
# no koordinaty sohranyayutsya v originalnom razreshenii.

import cv2
import numpy as np
import yaml
import argparse
import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Global state
points = []          # original resolution coordinates
display_scale = 1.0  # display_px / original_px
frame_orig = None    # full resolution frame
frame_display = None # resized frame for display


def click_event(event, x, y, flags, param):
    global points, frame_display, display_scale

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        # Map display coordinates back to original resolution
        ox = int(x / display_scale)
        oy = int(y / display_scale)
        points.append([ox, oy])
        print(f"  Tochka {len(points)}: ({ox}, {oy})  [display: ({x}, {y})]")

        # Draw on display frame (in display coordinates)
        r = max(4, int(6 / display_scale * display_scale))
        cv2.circle(frame_display, (x, y), r, (0, 255, 0), -1)
        font_scale = max(0.5, 0.7 * display_scale / 0.5)
        cv2.putText(frame_display, str(len(points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        if len(points) > 1:
            # Draw line in display coordinates
            prev_dx = int(points[-2][0] * display_scale)
            prev_dy = int(points[-2][1] * display_scale)
            cv2.line(frame_display, (prev_dx, prev_dy), (x, y), (0, 255, 0), 2)
        if len(points) == 4:
            p0_dx = int(points[0][0] * display_scale)
            p0_dy = int(points[0][1] * display_scale)
            cv2.line(frame_display, (x, y), (p0_dx, p0_dy), (0, 255, 0), 2)
            print("\n  4 tochki vybrany! Nazhmite lyubuyu klavishu...")

        cv2.imshow("Calibration", frame_display)


def main():
    global frame_display, frame_orig, points, display_scale

    parser = argparse.ArgumentParser(description="Calibration homography")
    parser.add_argument("--video", required=True, help="Path to video or image")
    parser.add_argument("--camera", type=str, default=None, help="Camera name")
    parser.add_argument("--output", default=None, help="Output file")
    parser.add_argument("--max-width", type=int, default=1280,
                        help="Max display width (default 1280)")
    args = parser.parse_args()

    if args.output is None:
        if args.camera:
            args.output = f"config/homography_{args.camera}.yaml"
        else:
            args.output = "homography_config.yaml"

    # Load frame
    if args.video.lower().endswith(('.jpg', '.png', '.jpeg')):
        frame_orig = cv2.imread(args.video)
    else:
        cap = cv2.VideoCapture(args.video)
        ret, frame_orig = cap.read()
        cap.release()
        if not ret:
            print("ERROR: cannot read video")
            return

    if frame_orig is None:
        print(f"ERROR: cannot read {args.video}")
        return

    h, w = frame_orig.shape[:2]

    # Scale down for display if too large
    if w > args.max_width:
        display_scale = args.max_width / w
    else:
        display_scale = 1.0

    disp_w = int(w * display_scale)
    disp_h = int(h * display_scale)

    if display_scale < 1.0:
        frame_display = cv2.resize(frame_orig, (disp_w, disp_h),
                                   interpolation=cv2.INTER_AREA)
    else:
        frame_display = frame_orig.copy()

    print("=" * 60)
    print("  KALIBROVKA GOMOGRAFII")
    print("=" * 60)
    print(f"  Resolution: {w}x{h}")
    if display_scale < 1.0:
        print(f"  Display: {disp_w}x{disp_h} (scale {display_scale:.2f})")
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
    print("  ESC = otmena, R = sbros tochek")
    print("=" * 60)

    cv2.imshow("Calibration", frame_display)
    cv2.setMouseCallback("Calibration", click_event)

    while len(points) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            print("Cancelled")
            cv2.destroyAllWindows()
            return
        elif key in (ord('r'), ord('R')):
            # Reset points
            points = []
            if display_scale < 1.0:
                frame_display = cv2.resize(frame_orig, (disp_w, disp_h),
                                           interpolation=cv2.INTER_AREA)
            else:
                frame_display = frame_orig.copy()
            cv2.imshow("Calibration", frame_display)
            print("  Tochki sbrosheny. Kliknite zanovo.")

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

    # Source points (original resolution pixels)
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
    print("  Source points (original pixels):")
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

    # Limit BEV display size
    max_bev_h = 800
    if bev_h > max_bev_h:
        bev_display_scale = max_bev_h / bev_h
        bev_w = int(bev_w * bev_display_scale)
        bev_h = max_bev_h
        scale_display = scale * bev_display_scale
    else:
        bev_display_scale = 1.0
        scale_display = scale

    dst_pts_shifted = dst_pts * bev_display_scale + np.array([int(50 * bev_display_scale), int(50 * bev_display_scale)])
    H_display, _ = cv2.findHomography(src_pts, dst_pts_shifted)

    bev = cv2.warpPerspective(frame_orig, H_display, (bev_w, bev_h))

    # Grid (every meter)
    margin = int(50 * bev_display_scale)
    for i in range(int(width_m) + 1):
        x = margin + int(i * scale_display)
        cv2.line(bev, (x, margin), (x, margin + int(length_m * scale_display)), (0, 255, 0), 1)
    for i in range(int(length_m) + 1):
        y = margin + int(i * scale_display)
        cv2.line(bev, (margin, y), (margin + int(width_m * scale_display), y), (0, 255, 0), 1)

    cv2.putText(bev, "Bird's Eye View (1 cell = 1m)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Bird's Eye View", bev)

    # Show original with points (scaled)
    orig_show = frame_display.copy()
    cv2.imshow("Original", orig_show)
    print("  Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
