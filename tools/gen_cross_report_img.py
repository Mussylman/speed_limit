#!/usr/bin/env python3
"""Generate a cross-camera speed report image with vehicle crops."""

import json
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def find_crop(report_vehicles_dir, plate_substr):
    """Find vehicle crop image matching plate substring."""
    if not os.path.isdir(report_vehicles_dir):
        return None
    for fn in sorted(os.listdir(report_vehicles_dir)):
        if plate_substr in fn and fn.endswith(".jpg"):
            return os.path.join(report_vehicles_dir, fn)
    return None


def draw_report(exp_dir, output_path=None):
    json_path = os.path.join(exp_dir, "cross_camera_speeds.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", "")
    speed_limit = data.get("speed_limit", 70)
    results = data.get("results", [])
    if not results:
        print("No results found.")
        return

    # Group by plate
    plates = {}
    for r in results:
        plates.setdefault(r["plate"], []).append(r)

    # --- Fonts ---
    try:
        font_title = ImageFont.truetype("arial.ttf", 32)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_header = ImageFont.truetype("arial.ttf", 16)
        font_cell = ImageFont.truetype("arial.ttf", 17)
        font_plate = ImageFont.truetype("arialbd.ttf", 36)
        font_speed_big = ImageFont.truetype("arialbd.ttf", 22)
        font_cam_label = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font_title = font_sub = font_header = font_cell = ImageFont.load_default()
        font_plate = font_speed_big = font_cam_label = font_title

    # --- Layout ---
    W = 1100
    PAD = 20
    TITLE_H = 80
    PLATE_H = 60
    TABLE_HEADER_H = 36
    ROW_H = 38
    CROP_H = 220
    CROP_PAD = 15
    SECTION_GAP = 30

    # Colors
    BG = (25, 25, 30)
    HEADER_BG = (35, 45, 60)
    PLATE_BG = (40, 40, 48)
    TH_BG = (50, 55, 65)
    ROW_BG1 = (35, 38, 45)
    ROW_BG2 = (42, 45, 52)
    WHITE = (255, 255, 255)
    GRAY = (170, 170, 180)
    GREEN = (80, 220, 100)
    RED = (240, 70, 70)
    YELLOW = (255, 210, 60)
    ACCENT = (100, 160, 255)

    # Calculate height
    total_h = TITLE_H + PAD
    for plate, rows in plates.items():
        total_h += PLATE_H + TABLE_HEADER_H + len(rows) * ROW_H + CROP_PAD + CROP_H + SECTION_GAP
    total_h += PAD

    img = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(img)

    # --- Title bar ---
    draw.rectangle([(0, 0), (W, TITLE_H)], fill=HEADER_BG)
    draw.text((PAD, 14), "Cross-Camera Speed Report", font=font_title, fill=YELLOW)
    draw.text((PAD, 52), name, font=font_sub, fill=GRAY)
    # Speed limit badge
    badge_text = f"Limit: {speed_limit} km/h"
    bbox = draw.textbbox((0, 0), badge_text, font=font_sub)
    bw = bbox[2] - bbox[0] + 20
    bx = W - bw - PAD
    draw.rounded_rectangle([(bx, 20), (bx + bw, 56)], radius=8, fill=(60, 30, 30))
    draw.text((bx + 10, 24), badge_text, font=font_sub, fill=RED)

    y = TITLE_H + PAD

    # Column positions
    cols = [PAD, 240, 400, 540, 640, 740, 900]
    headers = ["Segment", "Time From", "Time To", "Dist", "Delta", "Speed", "Avg"]

    for plate, rows in plates.items():
        # --- Plate header ---
        draw.rectangle([(0, y), (W, y + PLATE_H)], fill=PLATE_BG)
        if len(plate) == 8 and plate[:3].isdigit() and plate[6:].isdigit():
            plate_disp = f"{plate[:3]} {plate[3:6]} {plate[6:]}"
        else:
            plate_disp = plate
        draw.text((PAD, y + 10), plate_disp, font=font_plate, fill=WHITE)
        y += PLATE_H

        # --- Table header ---
        draw.rectangle([(0, y), (W, y + TABLE_HEADER_H)], fill=TH_BG)
        for i, hdr in enumerate(headers):
            draw.text((cols[i], y + 8), hdr, font=font_header, fill=ACCENT)
        y += TABLE_HEADER_H

        # --- Rows ---
        speeds = []
        for idx, r in enumerate(rows):
            bg = ROW_BG2 if idx % 2 else ROW_BG1
            draw.rectangle([(0, y), (W, y + ROW_H)], fill=bg)

            cam_from = r["cam_from"].replace("Camera_", "Cam ")
            cam_to = r["cam_to"].replace("Camera_", "Cam ")
            segment = f"{cam_from} → {cam_to}"
            t_from = r["time_from"].split("T")[1][:8]
            t_to = r["time_to"].split("T")[1][:8]
            dist = f"{r['distance_m']:.1f} m"
            delta = f"{r['time_sec']:.1f} s"
            speed = r["speed_kmh"]
            speed_str = f"{speed:.1f} km/h"
            spd_color = RED if speed > speed_limit else GREEN

            draw.text((cols[0], y + 9), segment, font=font_cell, fill=WHITE)
            draw.text((cols[1], y + 9), t_from, font=font_cell, fill=GRAY)
            draw.text((cols[2], y + 9), t_to, font=font_cell, fill=GRAY)
            draw.text((cols[3], y + 9), dist, font=font_cell, fill=WHITE)
            draw.text((cols[4], y + 9), delta, font=font_cell, fill=WHITE)
            draw.text((cols[5], y + 9), speed_str, font=font_speed_big, fill=spd_color)
            speeds.append(speed)
            y += ROW_H

        # Average row
        if speeds:
            avg = sum(speeds) / len(speeds)
            avg_str = f"≈ {avg:.1f} km/h"
            avg_color = RED if avg > speed_limit else GREEN
            # Draw on last row's avg column
            last_row_y = y - ROW_H
            draw.text((cols[6], last_row_y + 9), avg_str, font=font_speed_big, fill=avg_color)

        # --- Crop thumbnails ---
        y += CROP_PAD
        cam_ids = set()
        for r in rows:
            cam_ids.add(r["cam_from"])
            cam_ids.add(r["cam_to"])

        thumb_w = int(CROP_H * 0.95)
        tx = PAD
        for cam_id in sorted(cam_ids):
            cam_dir = os.path.join(exp_dir, cam_id, "report", "vehicles")
            crop_path = find_crop(cam_dir, plate[:6])  # match by first 6 chars
            if not crop_path:
                # Try alt plates that were merged
                crop_path = find_crop(cam_dir, plate[:3])
            if crop_path:
                crop_img = Image.open(crop_path)
                # Scale to fit
                scale = min(thumb_w / crop_img.width, CROP_H / crop_img.height)
                nw, nh = int(crop_img.width * scale), int(crop_img.height * scale)
                crop_resized = crop_img.resize((nw, nh), Image.LANCZOS)
                # Add border
                bordered = Image.new("RGB", (nw + 4, nh + 4), (80, 80, 90))
                bordered.paste(crop_resized, (2, 2))
                if tx + bordered.width < W - PAD:
                    img.paste(bordered, (tx, y))
                    # Camera label
                    label = cam_id.replace("Camera_", "Cam ")
                    draw.text((tx + 4, y + bordered.height + 2), label, font=font_cam_label, fill=GRAY)
                    tx += bordered.width + 12

        y += CROP_H + SECTION_GAP

    # Bottom border line
    draw.line([(0, total_h - 2), (W, total_h - 2)], fill=(80, 80, 90), width=2)

    out = output_path or os.path.join(exp_dir, "cross_camera_report_v2.jpg")
    img.save(out, quality=95)
    print(f"Report: {out}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    draw_report(args.exp, args.output)
