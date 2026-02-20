"""Batch OCR: чистый прогон NomeroffNet на папку car_crops → CSV.

Без фильтров, без majority vote — сырой результат пайплайна.

Usage:
    python tools/batch_ocr.py <folder> [--resize 0] [--csv out.csv]

Example:
    python tools/batch_ocr.py outputs/exp_61/Camera_14/debug_ocr/car_crops --csv cam14.csv
"""
import sys
import os
import re
import csv
import glob
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "nomeroff-net"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Regex for car_crop filenames: 00001_f419_t184_orig_350x317.jpg
CROP_RE = re.compile(
    r"(\d+)_f(\d+)_t(\d+)_orig_(\d+)x(\d+)\.(?:jpg|png)$", re.IGNORECASE
)


def parse_filename(fname):
    """Extract (seq, frame, track_id, orig_w, orig_h) from filename, or None."""
    m = CROP_RE.match(fname)
    if m:
        return int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])
    # Fallback: try to get at least frame and track
    parts = fname.split("_")
    frame, tid = 0, 0
    for p in parts:
        if p.startswith("f") and p[1:].isdigit():
            frame = int(p[1:])
        elif p.startswith("t") and p[1:].isdigit():
            tid = int(p[1:])
    return 0, frame, tid, 0, 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch OCR on car_crops → CSV")
    parser.add_argument("folder", help="Path to car_crops folder")
    parser.add_argument("--resize", type=int, default=0,
                        help="Resize width before OCR (0=no resize, default=0)")
    parser.add_argument("--csv", type=str, default="",
                        help="Output CSV path (default: <folder>/batch_ocr.csv)")
    args = parser.parse_args()

    images = sorted(
        glob.glob(os.path.join(args.folder, "*.jpg")) +
        glob.glob(os.path.join(args.folder, "*.png"))
    )
    if not images:
        print(f"No images found in {args.folder}")
        return

    csv_path = args.csv or os.path.join(args.folder, "batch_ocr.csv")

    print(f"Found {len(images)} images in {args.folder}")
    print(f"Resize: {args.resize}px" if args.resize else "No resize")
    print(f"CSV output: {csv_path}")
    print()

    # Load NomeroffNet
    print("Loading NomeroffNet...")
    from nomeroff_net import pipeline as nn_pipeline

    model_path = os.path.join(
        BASE_DIR, "nomeroff-net", "data", "models", "Detector",
        "yolov11x", "yolov11x-keypoints-2024-10-11.engine",
    )
    nomeroff = nn_pipeline(
        "number_plate_detection_and_reading_runtime",
        off_number_plate_classification=False,
        default_label="kz",
        default_lines_count=1,
        path_to_model=model_path,
    )
    print("NomeroffNet ready\n")

    from collections import defaultdict, Counter

    rows = []  # CSV rows
    tracks = defaultdict(list)

    HDR = f"{'#':>4} {'file':<45} {'frame':>5} {'tid':>4} {'crop_w':>6} {'text':<15} {'conf_min':>8} {'conf_avg':>8} {'plate_w':>7} {'det_conf':>8}"
    print(HDR)
    print("-" * len(HDR))

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        seq, frame, tid, orig_w, orig_h = parse_filename(fname)
        crop_w = orig_w if orig_w else img.shape[1]
        crop_h = orig_h if orig_h else img.shape[0]

        # Resize if requested
        feed = img
        if args.resize and img.shape[1] > args.resize:
            scale = args.resize / img.shape[1]
            feed = cv2.resize(img, (args.resize, int(img.shape[0] * scale)))

        rgb = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)
        try:
            result = nomeroff([rgb])
        except Exception as e:
            row = dict(file=fname, frame=frame, track_id=tid,
                       crop_w=crop_w, crop_h=crop_h,
                       text="ERROR", conf_min=0, conf_avg=0,
                       plate_w=0, det_conf=0)
            rows.append(row)
            print(f"{i+1:4d} {fname:<45} {frame:5d} {tid:4d} {crop_w:6d} {'ERROR':<15} ERROR: {e}")
            continue

        if not isinstance(result, (list, tuple)) or len(result) == 0:
            row = dict(file=fname, frame=frame, track_id=tid,
                       crop_w=crop_w, crop_h=crop_h,
                       text="NO_RESULT", conf_min=0, conf_avg=0,
                       plate_w=0, det_conf=0)
            rows.append(row)
            tracks[tid].append(row)
            print(f"{i+1:4d} {fname:<45} {frame:5d} {tid:4d} {crop_w:6d} {'NO_RESULT':<15}")
            continue

        data = result[0]
        if not isinstance(data, (list, tuple)) or len(data) < 9:
            row = dict(file=fname, frame=frame, track_id=tid,
                       crop_w=crop_w, crop_h=crop_h,
                       text="BAD_DATA", conf_min=0, conf_avg=0,
                       plate_w=0, det_conf=0)
            rows.append(row)
            tracks[tid].append(row)
            print(f"{i+1:4d} {fname:<45} {frame:5d} {tid:4d} {crop_w:6d} {'BAD_DATA':<15}")
            continue

        bboxs = data[1]
        texts = data[8]
        ocr_confs = data[7]

        if not bboxs or len(bboxs) == 0:
            row = dict(file=fname, frame=frame, track_id=tid,
                       crop_w=crop_w, crop_h=crop_h,
                       text="NO_PLATE", conf_min=0, conf_avg=0,
                       plate_w=0, det_conf=0)
            rows.append(row)
            tracks[tid].append(row)
            print(f"{i+1:4d} {fname:<45} {frame:5d} {tid:4d} {crop_w:6d} {'NO_PLATE':<15}")
            continue

        for j in range(len(bboxs)):
            bbox = bboxs[j]
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            pw = int(x2 - x1)
            det_conf = float(bbox[4]) if len(bbox) > 4 else 0.0

            text = ""
            if texts and j < len(texts):
                text = str(texts[j]).replace(" ", "").upper()

            conf_min, conf_avg = 0.0, 0.0
            if ocr_confs and j < len(ocr_confs):
                raw = ocr_confs[j]
                try:
                    if isinstance(raw, (list, tuple, np.ndarray)):
                        vals = [float(v) for v in raw if float(v) >= 0]
                        if vals:
                            conf_min = min(vals)
                            conf_avg = sum(vals) / len(vals)
                    elif isinstance(raw, (int, float)) and float(raw) >= 0:
                        conf_min = conf_avg = float(raw)
                except Exception:
                    pass

            row = dict(file=fname, frame=frame, track_id=tid,
                       crop_w=crop_w, crop_h=crop_h,
                       text=text, conf_min=round(conf_min, 4),
                       conf_avg=round(conf_avg, 4),
                       plate_w=pw, det_conf=round(det_conf, 4))
            rows.append(row)
            tracks[tid].append(row)
            print(f"{i+1:4d} {fname:<45} {frame:5d} {tid:4d} {crop_w:6d} {text:<15} {conf_min:8.4f} {conf_avg:8.4f} pw={pw:4d} det={det_conf:.3f}")

    # Write CSV
    fieldnames = ["file", "frame", "track_id", "crop_w", "crop_h",
                  "text", "conf_min", "conf_avg", "plate_w", "det_conf"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path} ({len(rows)} rows)")

    # Summary by track
    print(f"\n{'=' * 95}")
    print("SUMMARY BY TRACK")
    print(f"{'=' * 95}")
    for tid in sorted(tracks.keys()):
        items = tracks[tid]
        valid = [r for r in items if r["text"] not in ("NO_PLATE", "NO_RESULT", "BAD_DATA", "ERROR", "")]
        no_plate = len(items) - len(valid)
        print(f"\n  Track t{tid}: {len(items)} images, {no_plate} no_plate")

        counter = Counter(r["text"] for r in valid)
        for text, count in counter.most_common():
            matches = [r for r in valid if r["text"] == text]
            avg_min = sum(r["conf_min"] for r in matches) / count
            avg_avg = sum(r["conf_avg"] for r in matches) / count
            print(f"    {text:<15} x{count:2d}  conf_min_avg={avg_min:.3f}  conf_avg_avg={avg_avg:.3f}")


if __name__ == "__main__":
    main()
