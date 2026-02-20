#!/usr/bin/env python3
"""
Cross-camera speed report generator.

Collects results.json from all camera dirs in an experiment,
matches plates across cameras, calculates average speeds between
camera pairs, and generates a summary image + JSON.

Usage:
    python tools/cross_camera_report.py --exp outputs/exp_17 --name "15 km/h test"
    python tools/cross_camera_report.py --dirs outputs/exp_15/Camera_12 outputs/exp_16/Camera_14 outputs/exp_17/Camera_21 --name "15 km/h"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import BASE_DIR


def load_config():
    """Load cross-camera distances from config_cam.yaml."""
    import yaml
    cfg_path = os.path.join(BASE_DIR, "config", "config_cam.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cross = cfg.get("cross_camera", {})
    distances = {}
    for entry in cross.get("distances_m", []):
        cams = entry["cameras"]
        dist = float(entry["distance"])
        distances[(cams[0], cams[1])] = dist
        distances[(cams[1], cams[0])] = dist
    # Auto-sum indirect (A→C = A→B + B→C)
    all_cams = list({c for pair in distances for c in pair})
    for ca in all_cams:
        for cc in all_cams:
            if ca == cc or (ca, cc) in distances:
                continue
            for cb in all_cams:
                if cb in (ca, cc):
                    continue
                d1 = distances.get((ca, cb))
                d2 = distances.get((cb, cc))
                if d1 and d2:
                    distances[(ca, cc)] = d1 + d2
                    distances[(cc, ca)] = d1 + d2
    speed_limit = cross.get("speed_limit", 70)
    return distances, speed_limit


def _load_alt_plates(cam_dir, track_id):
    """Load alternative plate readings from res_ocr/ for a track."""
    res_ocr_dir = os.path.join(cam_dir, "res_ocr")
    if not os.path.isdir(res_ocr_dir) or track_id is None:
        return []
    suffix = f"_id{track_id}.json"
    plates = set()
    for fn in os.listdir(res_ocr_dir):
        if fn.endswith(suffix):
            try:
                with open(os.path.join(res_ocr_dir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                pt = data.get("plate_text", "")
                if pt and pt != "UNKNOWN":
                    plates.add(pt)
            except (json.JSONDecodeError, OSError):
                pass
    return list(plates)


def collect_results(dirs):
    """Load results.json from each camera directory."""
    all_events = []
    for d in dirs:
        rpath = os.path.join(d, "passed", "results.json")
        if not os.path.isfile(rpath):
            continue
        with open(rpath, "r", encoding="utf-8") as f:
            events = json.load(f)
        for ev in events:
            ev["_dir"] = d
            ev["_alt_plates"] = _load_alt_plates(d, ev.get("track_id"))
        all_events.extend(events)
    return all_events


def match_plates(events, distances):
    """Group events by plate, compute cross-camera speeds.

    Two-pass matching:
    1. Exact + ocr_variants (single-char substitution)
    2. Levenshtein fallback using all res_ocr readings for unmatched single-camera plates
    """
    from kz_plate import ocr_variants, ocr_distance

    # --- Pass 1: exact + variant matching ---
    plate_groups = defaultdict(list)
    canonical = {}  # map variant -> canonical plate

    for ev in events:
        plate = ev["plate_text"]
        if plate == "UNKNOWN":
            continue
        # Check if any variant matches existing canonical
        matched = None
        for v in ocr_variants(plate):
            if v in canonical:
                matched = canonical[v]
                break
        if matched:
            canonical[plate] = matched
            plate_groups[matched].append(ev)
        else:
            canonical[plate] = plate
            plate_groups[plate].append(ev)

    # --- Pass 2: Levenshtein merge for single-camera groups ---
    # Find groups seen on only one camera
    single_cam = {}
    multi_cam = {}
    for plate, evs in plate_groups.items():
        cams = {e["camera_id"] for e in evs}
        if len(cams) == 1:
            single_cam[plate] = evs
        else:
            multi_cam[plate] = evs

    OCR_DIST_THRESHOLD = 2.0
    merged = set()
    for s_plate, s_evs in single_cam.items():
        if s_plate in merged:
            continue
        # Collect all OCR variants from res_ocr for this plate's tracks
        all_texts = {s_plate}
        for ev in s_evs:
            all_texts.update(ev.get("_alt_plates", []))

        # Try matching against multi-camera groups first
        best_match = None
        best_dist = OCR_DIST_THRESHOLD + 1
        for alt in all_texts:
            for m_plate in multi_cam:
                d = ocr_distance(alt, m_plate)
                if d < best_dist:
                    best_dist = d
                    best_match = m_plate

        # Then try matching against other single-camera groups (different camera)
        if best_match is None or best_dist > OCR_DIST_THRESHOLD:
            s_cam = next(iter({e["camera_id"] for e in s_evs}))
            for s2_plate, s2_evs in single_cam.items():
                if s2_plate == s_plate or s2_plate in merged:
                    continue
                s2_cam = next(iter({e["camera_id"] for e in s2_evs}))
                if s2_cam == s_cam:
                    continue
                s2_texts = {s2_plate}
                for ev in s2_evs:
                    s2_texts.update(ev.get("_alt_plates", []))
                for a1 in all_texts:
                    for a2 in s2_texts:
                        d = ocr_distance(a1, a2)
                        if d < best_dist:
                            best_dist = d
                            best_match = s2_plate

        if best_match and best_dist <= OCR_DIST_THRESHOLD:
            # Merge into the target group
            target = multi_cam.get(best_match, plate_groups.get(best_match))
            target.extend(s_evs)
            if best_match not in multi_cam:
                multi_cam[best_match] = target
            merged.add(s_plate)
            print(f"  [CROSS-CAM] Levenshtein merge: {s_plate} -> {best_match} (dist={best_dist})")

    # Remove merged groups
    for m in merged:
        del plate_groups[m]

    results = []
    for plate, evs in plate_groups.items():
        if len(evs) < 2:
            continue
        # Sort by timestamp
        evs.sort(key=lambda e: e["timestamp"])
        # Compute speeds for all pairs
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                e1, e2 = evs[i], evs[j]
                c1, c2 = e1["camera_id"], e2["camera_id"]
                if c1 == c2:
                    continue
                dist = distances.get((c1, c2))
                if dist is None:
                    continue
                t1 = datetime.fromisoformat(e1["timestamp"])
                t2 = datetime.fromisoformat(e2["timestamp"])
                dt = (t2 - t1).total_seconds()
                if dt < 0.5:
                    continue
                speed_kmh = (dist / dt) * 3.6
                results.append({
                    "plate": plate,
                    "cam_from": c1,
                    "cam_to": c2,
                    "label_from": e1.get("camera_label", c1),
                    "label_to": e2.get("camera_label", c2),
                    "time_from": e1["timestamp"],
                    "time_to": e2["timestamp"],
                    "distance_m": dist,
                    "time_sec": round(dt, 1),
                    "speed_kmh": round(speed_kmh, 1),
                    "crop_from": os.path.join(e1["_dir"], "passed", e1.get("crop_path", "")),
                    "crop_to": os.path.join(e2["_dir"], "passed", e2.get("crop_path", "")),
                })
    return results


# ── Drawing ──────────────────────────────────────────────────────

BG_DARK = (30, 30, 30)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 220, 255)
HEADER_BG = (45, 45, 45)
ROW_BG = (40, 40, 40)
ROW_ALT = (50, 50, 50)
BORDER = (80, 80, 80)


def put_text_pil(img, text, pos, font_size=20, color=(255, 255, 255)):
    """Draw text via PIL for Unicode support. Color is BGR."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    rgb = (color[2], color[1], color[0])
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=rgb)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)


def draw_report(results, name, speed_limit, output_path):
    """Draw the cross-camera summary image."""

    # Group by plate
    plates = defaultdict(list)
    for r in results:
        plates[r["plate"]].append(r)

    W = 1000
    title_h = 70
    plate_header_h = 50
    row_h = 40
    table_header_h = 35
    pad = 15
    crop_h = 140

    # Calculate total height
    total_h = title_h + pad
    for plate, rows in plates.items():
        total_h += plate_header_h + table_header_h + len(rows) * row_h + crop_h + pad * 2

    img = np.full((total_h, W, 3), BG_DARK, dtype=np.uint8)

    # ── Title ──
    cv2.rectangle(img, (0, 0), (W, title_h), HEADER_BG, -1)
    put_text_pil(img, f"Cross-Camera Speed Report", (15, 8), 26, YELLOW)
    put_text_pil(img, name, (15, 40), 18, GRAY)

    y = title_h + pad

    col_x = [15, 230, 420, 570, 690, 830]  # segment, time_from, time_to, dist, delta, speed

    for plate, rows in plates.items():
        # ── Plate header ──
        plate_display = plate
        if len(plate) == 8 and plate[:3].isdigit() and plate[6:].isdigit():
            plate_display = f"{plate[:3]} {plate[3:6]} {plate[6:]}"

        cv2.rectangle(img, (0, y), (W, y + plate_header_h), (55, 55, 55), -1)
        cv2.putText(img, plate_display, (15, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
        y += plate_header_h

        # ── Table header ──
        cv2.rectangle(img, (0, y), (W, y + table_header_h), (60, 60, 60), -1)
        headers = ["Segment", "Time From", "Time To", "Dist", "Delta", "Speed"]
        for i, hdr in enumerate(headers):
            put_text_pil(img, hdr, (col_x[i], y + 8), 14, GRAY)
        y += table_header_h

        # ── Data rows ──
        for idx, r in enumerate(rows):
            bg = ROW_ALT if idx % 2 else ROW_BG
            cv2.rectangle(img, (0, y), (W, y + row_h), bg, -1)

            segment = f"{r['label_from']} -> {r['label_to']}"
            t_from = r["time_from"].split("T")[1][:12]
            t_to = r["time_to"].split("T")[1][:12]
            dist = f"{r['distance_m']:.1f} m"
            delta = f"{r['time_sec']:.1f} s"
            speed = f"{r['speed_kmh']:.1f} km/h"
            spd_color = RED if r["speed_kmh"] > speed_limit else GREEN

            put_text_pil(img, segment, (col_x[0], y + 10), 15, WHITE)
            put_text_pil(img, t_from, (col_x[1], y + 10), 15, WHITE)
            put_text_pil(img, t_to, (col_x[2], y + 10), 15, WHITE)
            put_text_pil(img, dist, (col_x[3], y + 10), 15, WHITE)
            put_text_pil(img, delta, (col_x[4], y + 10), 15, WHITE)
            put_text_pil(img, speed, (col_x[5], y + 10), 16, spd_color)

            y += row_h

        # ── Crop thumbnails ──
        y += 5
        # Collect unique camera crops for this plate
        seen_crops = {}
        for r in rows:
            for key, path in [("from", r["crop_from"]), ("to", r["crop_to"])]:
                cam = r[f"cam_{key}"]
                if cam not in seen_crops and os.path.isfile(path):
                    seen_crops[cam] = path

        thumb_w = int(crop_h * 1.3)
        tx = 15
        for cam, crop_path in seen_crops.items():
            crop_img = cv2.imread(crop_path)
            if crop_img is None:
                continue
            h_c, w_c = crop_img.shape[:2]
            scale = min(thumb_w / w_c, crop_h / h_c)
            nw, nh = int(w_c * scale), int(h_c * scale)
            resized = cv2.resize(crop_img, (nw, nh))
            # Center in thumb area
            y_off = (crop_h - nh) // 2
            x_off = (thumb_w - nw) // 2
            if y + y_off + nh <= img.shape[0] and tx + x_off + nw <= img.shape[1]:
                img[y + y_off:y + y_off + nh, tx + x_off:tx + x_off + nw] = resized
                # Camera label under thumbnail
                label = cam.replace("Camera_", "Cam ")
                put_text_pil(img, label, (tx + 5, y + crop_h - 18), 12, GRAY)
            tx += thumb_w + 10

        y += crop_h + pad

    # Border
    cv2.rectangle(img, (0, 0), (W - 1, total_h - 1), BORDER, 2)

    cv2.imwrite(output_path, img)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Cross-camera speed report")
    parser.add_argument("--exp", help="Experiment root dir (auto-finds Camera_* subdirs)")
    parser.add_argument("--dirs", nargs="+", help="Explicit camera result dirs")
    parser.add_argument("--name", default="", help="Report title / test name")
    parser.add_argument("--output", help="Output image path (default: <exp>/cross_camera_report.jpg)")
    args = parser.parse_args()

    # Determine camera dirs
    if args.exp:
        exp_dir = args.exp
        cam_dirs = []
        for entry in sorted(os.listdir(exp_dir)):
            full = os.path.join(exp_dir, entry)
            if os.path.isdir(full) and entry.startswith("Camera_"):
                cam_dirs.append(full)
    elif args.dirs:
        cam_dirs = args.dirs
        exp_dir = os.path.dirname(cam_dirs[0]) if cam_dirs else "."
    else:
        print("Specify --exp or --dirs")
        sys.exit(1)

    if not cam_dirs:
        print(f"No Camera_* dirs found in {exp_dir}")
        sys.exit(1)

    print(f"Cameras: {[os.path.basename(d) for d in cam_dirs]}")

    distances, speed_limit = load_config()
    events = collect_results(cam_dirs)
    print(f"Loaded {len(events)} plate events")

    results = match_plates(events, distances)
    if not results:
        print("No cross-camera matches found.")
        sys.exit(0)

    # Print table
    print(f"\n{'Plate':<12} {'Segment':<30} {'Time From':<16} {'Time To':<16} "
          f"{'Dist':>7} {'Delta':>7} {'Speed':>10}")
    print("-" * 105)
    for r in results:
        marker = " !!!" if r["speed_kmh"] > speed_limit else ""
        print(f"{r['plate']:<12} {r['cam_from']}->{r['cam_to']:<20} "
              f"{r['time_from'].split('T')[1]:<16} {r['time_to'].split('T')[1]:<16} "
              f"{r['distance_m']:>6.1f}m {r['time_sec']:>6.1f}s "
              f"{r['speed_kmh']:>8.1f} km/h{marker}")

    # Save JSON
    out_dir = args.output and os.path.dirname(args.output) or exp_dir
    json_path = os.path.join(out_dir, "cross_camera_speeds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "speed_limit": speed_limit,
                    "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    # Draw image
    img_path = args.output or os.path.join(exp_dir, "cross_camera_report.jpg")
    draw_report(results, args.name or os.path.basename(exp_dir), speed_limit, img_path)
    print(f"Image: {img_path}")


if __name__ == "__main__":
    main()
