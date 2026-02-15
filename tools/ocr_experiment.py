"""
OCR Experiment Tool — measure OCR success/failure vs car/plate sizes.

Processes test videos frame-by-frame with YOLO tracking + PlateRecognizer,
logging every detection's metrics. Generates CSVs, summary, and plots.

Usage:
    python tools/ocr_experiment.py
    python tools/ocr_experiment.py --videos videos/test_1.mp4 videos/test_2.mp4
"""

import argparse
import csv
import copy
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nomeroff-net"))
from config import BASE_DIR, load_yaml, CONFIG_PATH

# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "video", "vid_width", "vid_height", "frame_idx", "track_id",
    "det_conf", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "car_width", "car_height", "car_area", "car_y_center_pct",
    "blur", "brightness", "quality_score",
    "ocr_ran",
    "plate_width", "plate_height", "plate_conf", "ocr_conf",
    "plate_text", "text_length", "format_valid", "region",
    "result", "reject_detail", "processing_ms",
]

RESULT_TYPES = [
    "passed", "skipped_car_size", "skipped_quality", "skipped_not_better",
    "skipped_cooldown", "skipped_no_plate_cooldown", "skipped_plate_size",
    "ocr_no_plate", "skipped_low_conf", "skipped_chars", "skipped_format",
    "unconfirmed",
]


def detect_video_files() -> list:
    """Find all test_*.mp4 videos in videos/."""
    vdir = os.path.join(BASE_DIR, "videos")
    if not os.path.isdir(vdir):
        return []
    files = sorted(
        f for f in os.listdir(vdir)
        if f.startswith("test_") and f.endswith(".mp4")
    )
    return [os.path.join(vdir, f) for f in files]


def load_models(cfg: dict):
    """Load YOLO and PlateRecognizer once."""
    from ultralytics import YOLO
    from plate_recognizer import PlateRecognizer

    yolo_path = os.path.join(BASE_DIR, cfg["models"]["yolo_model"])
    print(f"Loading YOLO from {yolo_path}...")
    yolo = YOLO(yolo_path, task="detect")

    device = cfg.get("device", "cuda")
    try:
        yolo.to(device)
    except Exception:
        device = "cpu"

    import torch
    use_half = device == "cuda" and torch.cuda.is_available()

    # Create recognizer with config values (no file_logger needed)
    output_tmp = os.path.join(BASE_DIR, "ocr_experiments", "_tmp_recognizer")
    recognizer = PlateRecognizer(
        output_dir=output_tmp,
        camera_id="experiment",
        min_conf=float(cfg.get("ocr_conf_threshold", 0.5)),
        min_plate_chars=int(cfg.get("min_plate_chars", 8)),
        min_car_height=int(cfg.get("min_car_height", 40)),
        min_car_width=int(cfg.get("min_car_width", 30)),
        min_plate_width=int(cfg.get("min_plate_width", 40)),
        min_plate_height=int(cfg.get("min_plate_height", 10)),
        cooldown_frames=int(cfg.get("ocr_cooldown_frames", 1)),
        plate_format_regex=cfg.get("plate_format_regex", ""),
        min_blur_score=float(cfg.get("_min_blur_score", 5.0)),
        min_brightness=float(cfg.get("_min_brightness", 20.0)),
        max_brightness=float(cfg.get("_max_brightness", 240.0)),
        quality_improvement=float(cfg.get("_quality_improvement", 1.0)),
        min_confirmations=int(cfg.get("_min_confirmations", 2)),
    )

    return yolo, recognizer, use_half, device


def reset_recognizer(recognizer):
    """Clear per-video state in recognizer so videos don't bleed into each other."""
    recognizer.passed_results.clear()
    recognizer.failed_results.clear()
    recognizer.pending_results.clear()
    recognizer.best_quality.clear()
    recognizer.last_ocr_frame.clear()
    recognizer.text_votes.clear()
    recognizer.char_history.clear()
    recognizer.full_frames.clear()
    recognizer.no_plate_count.clear()
    recognizer.no_plate_until.clear()
    for k in recognizer.stats:
        recognizer.stats[k] = 0


def snapshot_stats(recognizer) -> dict:
    """Return a shallow copy of recognizer.stats."""
    return dict(recognizer.stats)


def diff_stats(before: dict, after: dict) -> str:
    """Return the stat key that incremented (the filter that triggered)."""
    for key in [
        "skipped_car_size", "skipped_quality", "skipped_not_better",
        "skipped_cooldown", "skipped_no_plate_cooldown", "ocr_no_plate",
        "skipped_plate_size", "skipped_low_conf", "skipped_chars",
        "skipped_format", "unconfirmed", "passed",
    ]:
        if after.get(key, 0) > before.get(key, 0):
            return key
    # ocr_called went up but nothing else → shouldn't happen, but fallback
    if after.get("ocr_called", 0) > before.get("ocr_called", 0):
        return "ocr_called_unknown"
    return "unknown"


def process_video(video_path: str, yolo, recognizer, use_half: bool,
                  device: str, cfg: dict, out_dir: str) -> list:
    """Process one video, return list of row dicts."""
    import re

    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name} ({video_path})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return []

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  Resolution: {vid_w}x{vid_h}, FPS: {fps:.1f}, Frames: {total_frames}")

    # Reset recognizer state + YOLO tracker
    reset_recognizer(recognizer)
    yolo.predictor = None  # forces tracker reset on next .track() call

    imgsz = int(cfg.get("yolo_imgsz", 960))
    plate_regex = cfg.get("plate_format_regex", "")

    rows = []
    frame_idx = 0
    t_start = time.time()
    last_report = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tracking
        results = yolo.track(
            frame, imgsz=imgsz, classes=[2], persist=True,
            verbose=False, half=use_half, tracker="bytetrack.yaml",
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(ids)):
                track_id = int(ids[i])
                x1, y1, x2, y2 = int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])
                det_conf = float(confs[i])
                car_w = x2 - x1
                car_h = y2 - y1
                car_area = car_w * car_h
                car_y_center_pct = ((y1 + y2) / 2) / vid_h

                # Crop
                cx1 = max(0, x1)
                cy1 = max(0, y1)
                cx2 = min(vid_w, x2)
                cy2 = min(vid_h, y2)
                car_crop = frame[cy1:cy2, cx1:cx2]

                # Calculate quality externally (gives blur/brightness even for skipped)
                quality_score, blur, brightness = recognizer._calculate_quality(
                    car_crop, car_w, car_h
                )

                # Snapshot stats before process()
                stats_before = snapshot_stats(recognizer)

                t_proc = time.time()
                result_event = recognizer.process(
                    track_id=track_id,
                    car_crop=car_crop,
                    car_height=car_h,
                    car_width=car_w,
                    frame_idx=frame_idx,
                    detection_conf=det_conf,
                )
                proc_ms = (time.time() - t_proc) * 1000

                # Determine result via stats diff
                stats_after = snapshot_stats(recognizer)
                result_type = diff_stats(stats_before, stats_after)

                # Did OCR actually run?
                ocr_ran = stats_after.get("ocr_called", 0) > stats_before.get("ocr_called", 0)

                # Extract plate data from result_event or recognizer dicts
                plate_w = 0
                plate_h = 0
                plate_conf = 0.0
                ocr_conf = 0.0
                plate_text = ""
                text_len = 0
                format_valid = False
                region = ""
                reject_detail = ""

                # Check for data in various sources
                evt = None
                if track_id in recognizer.passed_results:
                    evt = recognizer.passed_results[track_id]
                elif track_id in recognizer.pending_results:
                    evt = recognizer.pending_results[track_id]
                elif track_id in recognizer.failed_results:
                    evt = recognizer.failed_results[track_id]

                if evt and evt.frame_idx == frame_idx:
                    # This event was just created/updated for this frame
                    plate_w = evt.plate_width_px
                    plate_h = evt.plate_height_px
                    plate_conf = evt.plate_conf
                    ocr_conf = evt.ocr_conf
                    plate_text = evt.plate_text
                    text_len = len(plate_text)
                    region = evt.region
                    reject_detail = evt.reject_reason
                    if plate_regex and plate_text:
                        format_valid = bool(re.match(plate_regex, plate_text))

                row = {
                    "video": video_name,
                    "vid_width": vid_w,
                    "vid_height": vid_h,
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "det_conf": round(det_conf, 4),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "car_width": car_w,
                    "car_height": car_h,
                    "car_area": car_area,
                    "car_y_center_pct": round(car_y_center_pct, 4),
                    "blur": round(blur, 1),
                    "brightness": round(brightness, 1),
                    "quality_score": round(quality_score, 2),
                    "ocr_ran": int(ocr_ran),
                    "plate_width": plate_w,
                    "plate_height": plate_h,
                    "plate_conf": round(plate_conf, 4),
                    "ocr_conf": round(ocr_conf, 4),
                    "plate_text": plate_text,
                    "text_length": text_len,
                    "format_valid": int(format_valid),
                    "region": region,
                    "result": result_type,
                    "reject_detail": reject_detail,
                    "processing_ms": round(proc_ms, 1),
                }
                rows.append(row)

        frame_idx += 1

        # Progress report every 500 frames
        if frame_idx - last_report >= 500:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%) | "
                  f"{fps_proc:.1f} fps | detections: {len(rows)}")
            last_report = frame_idx

    cap.release()
    elapsed = time.time() - t_start
    print(f"  Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps), "
          f"{len(rows)} detections")

    # Write per-video CSV
    csv_path = os.path.join(out_dir, f"{video_name}_details.csv")
    write_csv(csv_path, rows)
    print(f"  Saved: {csv_path}")

    return rows


def write_csv(path: str, rows: list):
    """Write rows to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def generate_summary(all_rows: dict, out_dir: str):
    """Generate summary.csv with per-video aggregates."""
    summary_rows = []
    for video_name, rows in all_rows.items():
        total = len(rows)
        if total == 0:
            continue
        ocr_ran = sum(1 for r in rows if r["ocr_ran"])
        passed = sum(1 for r in rows if r["result"] == "passed")

        # Count each result type
        result_counts = defaultdict(int)
        for r in rows:
            result_counts[r["result"]] += 1

        # Car height stats
        car_heights = [r["car_height"] for r in rows]
        # Plate widths (only where OCR ran)
        plate_widths = [r["plate_width"] for r in rows if r["ocr_ran"] and r["plate_width"] > 0]

        summary = {
            "video": video_name,
            "vid_width": rows[0]["vid_width"],
            "vid_height": rows[0]["vid_height"],
            "total_detections": total,
            "ocr_ran": ocr_ran,
            "ocr_ran_pct": round(ocr_ran / total * 100, 1) if total else 0,
            "passed": passed,
            "pass_rate_pct": round(passed / total * 100, 1) if total else 0,
            "pass_rate_of_ocr_pct": round(passed / ocr_ran * 100, 1) if ocr_ran else 0,
            "avg_car_height": round(np.mean(car_heights), 1),
            "median_car_height": round(np.median(car_heights), 1),
            "avg_plate_width": round(np.mean(plate_widths), 1) if plate_widths else 0,
            "median_plate_width": round(np.median(plate_widths), 1) if plate_widths else 0,
            "unique_tracks": len(set(r["track_id"] for r in rows)),
        }
        # Add per-result counts
        for rt in RESULT_TYPES:
            summary[f"n_{rt}"] = result_counts.get(rt, 0)

        summary_rows.append(summary)

    if not summary_rows:
        return

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved: {csv_path}")


def generate_report(all_rows: dict, out_dir: str):
    """Generate cross-video text report."""
    report_path = os.path.join(out_dir, "cross_video_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("OCR EXPERIMENT — CROSS-VIDEO REPORT")
    lines.append("=" * 70)

    total_all = sum(len(rows) for rows in all_rows.values())
    ocr_all = sum(1 for rows in all_rows.values() for r in rows if r["ocr_ran"])
    passed_all = sum(1 for rows in all_rows.values() for r in rows if r["result"] == "passed")

    lines.append(f"\nTotal detections: {total_all}")
    lines.append(f"OCR ran: {ocr_all} ({ocr_all/total_all*100:.1f}%)" if total_all else "OCR ran: 0")
    lines.append(f"Passed: {passed_all} ({passed_all/total_all*100:.1f}%)" if total_all else "Passed: 0")

    lines.append(f"\n{'Video':<15} {'Res':<12} {'Dets':>6} {'OCR':>6} {'Pass':>6} {'Rate':>7}")
    lines.append("-" * 60)
    for video_name, rows in all_rows.items():
        total = len(rows)
        ocr = sum(1 for r in rows if r["ocr_ran"])
        passed = sum(1 for r in rows if r["result"] == "passed")
        rate = f"{passed/total*100:.1f}%" if total else "0%"
        res = f"{rows[0]['vid_width']}x{rows[0]['vid_height']}" if rows else "?"
        lines.append(f"{video_name:<15} {res:<12} {total:>6} {ocr:>6} {passed:>6} {rate:>7}")

    # Result type breakdown
    lines.append(f"\n{'Result Type':<25} {'Count':>8} {'Pct':>7}")
    lines.append("-" * 45)
    all_flat = [r for rows in all_rows.values() for r in rows]
    result_counts = defaultdict(int)
    for r in all_flat:
        result_counts[r["result"]] += 1
    for rt in RESULT_TYPES:
        cnt = result_counts.get(rt, 0)
        pct = f"{cnt/total_all*100:.1f}%" if total_all else "0%"
        lines.append(f"{rt:<25} {cnt:>8} {pct:>7}")

    # Car height analysis
    lines.append("\n\nCAR HEIGHT ANALYSIS (pass rate by 20px bin):")
    lines.append(f"{'Bin':<15} {'Total':>6} {'Passed':>6} {'Rate':>7}")
    lines.append("-" * 40)
    heights = [(r["car_height"], r["result"]) for r in all_flat]
    if heights:
        max_h = max(h for h, _ in heights)
        for bin_start in range(0, int(max_h) + 20, 20):
            bin_end = bin_start + 20
            in_bin = [(h, res) for h, res in heights if bin_start <= h < bin_end]
            if not in_bin:
                continue
            passed_in_bin = sum(1 for _, res in in_bin if res == "passed")
            rate = f"{passed_in_bin/len(in_bin)*100:.1f}%"
            lines.append(f"{bin_start:>3}-{bin_end:<10} {len(in_bin):>6} {passed_in_bin:>6} {rate:>7}")

    # Plate width analysis (OCR-ran only)
    lines.append("\n\nPLATE WIDTH ANALYSIS (pass rate by 10px bin, OCR-ran only):")
    lines.append(f"{'Bin':<15} {'Total':>6} {'Passed':>6} {'Rate':>7}")
    lines.append("-" * 40)
    pw_data = [(r["plate_width"], r["result"]) for r in all_flat if r["ocr_ran"] and r["plate_width"] > 0]
    if pw_data:
        max_pw = max(w for w, _ in pw_data)
        for bin_start in range(0, int(max_pw) + 10, 10):
            bin_end = bin_start + 10
            in_bin = [(w, res) for w, res in pw_data if bin_start <= w < bin_end]
            if not in_bin:
                continue
            passed_in_bin = sum(1 for _, res in in_bin if res == "passed")
            rate = f"{passed_in_bin/len(in_bin)*100:.1f}%"
            lines.append(f"{bin_start:>3}-{bin_end:<10} {len(in_bin):>6} {passed_in_bin:>6} {rate:>7}")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")
    print(report_text)


def generate_plots(all_rows: dict, out_dir: str):
    """Generate analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_flat = [r for rows in all_rows.values() for r in rows]
    if not all_flat:
        return

    # Color map for result types
    color_map = {
        "passed": "#2ecc71",
        "skipped_car_size": "#e74c3c",
        "skipped_quality": "#e67e22",
        "skipped_not_better": "#f39c12",
        "skipped_cooldown": "#9b59b6",
        "skipped_plate_size": "#3498db",
        "ocr_no_plate": "#1abc9c",
        "skipped_low_conf": "#e91e63",
        "skipped_chars": "#795548",
        "skipped_format": "#607d8b",
        "unconfirmed": "#ff9800",
    }

    # 1. Car height distribution (stacked histogram by result type)
    print("  Plotting car_height_distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    max_h = max(r["car_height"] for r in all_flat)
    bins = np.arange(0, max_h + 20, 20)
    data_by_result = {}
    for rt in RESULT_TYPES:
        vals = [r["car_height"] for r in all_flat if r["result"] == rt]
        if vals:
            data_by_result[rt] = vals
    ax.hist(
        list(data_by_result.values()),
        bins=bins, stacked=True, label=list(data_by_result.keys()),
        color=[color_map.get(rt, "#999") for rt in data_by_result.keys()],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Car Height (px)")
    ax.set_ylabel("Count")
    ax.set_title("Car Height Distribution by Result Type")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "car_height_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Plate width distribution (OCR-ran detections only)
    print("  Plotting plate_width_distribution...")
    ocr_rows = [r for r in all_flat if r["ocr_ran"] and r["plate_width"] > 0]
    if ocr_rows:
        fig, ax = plt.subplots(figsize=(12, 6))
        max_pw = max(r["plate_width"] for r in ocr_rows)
        bins = np.arange(0, max_pw + 10, 10)
        data_by_result = {}
        for rt in RESULT_TYPES:
            vals = [r["plate_width"] for r in ocr_rows if r["result"] == rt]
            if vals:
                data_by_result[rt] = vals
        ax.hist(
            list(data_by_result.values()),
            bins=bins, stacked=True, label=list(data_by_result.keys()),
            color=[color_map.get(rt, "#999") for rt in data_by_result.keys()],
            edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Plate Width (px)")
        ax.set_ylabel("Count")
        ax.set_title("Plate Width Distribution by Result Type (OCR-ran only)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "plate_width_distribution.png"), dpi=150)
        plt.close(fig)

    # 3. Pass rate by car height
    print("  Plotting pass_rate_by_car_height...")
    fig, ax = plt.subplots(figsize=(12, 6))
    max_h = max(r["car_height"] for r in all_flat)
    bin_edges = list(range(0, int(max_h) + 20, 20))
    bin_centers = []
    pass_rates = []
    bin_counts = []
    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [r for r in all_flat if lo <= r["car_height"] < hi]
        if len(in_bin) < 3:
            continue
        passed = sum(1 for r in in_bin if r["result"] == "passed")
        bin_centers.append((lo + hi) / 2)
        pass_rates.append(passed / len(in_bin) * 100)
        bin_counts.append(len(in_bin))
    ax.bar(bin_centers, pass_rates, width=18, color="#2ecc71", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Car Height (px)")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate by Car Height (20px bins, min 3 samples)")
    # Add count labels
    for x, rate, cnt in zip(bin_centers, pass_rates, bin_counts):
        ax.text(x, rate + 1, f"n={cnt}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pass_rate_by_car_height.png"), dpi=150)
    plt.close(fig)

    # 4. Pass rate by plate width (OCR-ran only)
    print("  Plotting pass_rate_by_plate_width...")
    if ocr_rows:
        fig, ax = plt.subplots(figsize=(12, 6))
        max_pw = max(r["plate_width"] for r in ocr_rows)
        bin_edges = list(range(0, int(max_pw) + 10, 10))
        bin_centers = []
        pass_rates = []
        bin_counts = []
        for b in range(len(bin_edges) - 1):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_bin = [r for r in ocr_rows if lo <= r["plate_width"] < hi]
            if len(in_bin) < 3:
                continue
            passed = sum(1 for r in in_bin if r["result"] == "passed")
            bin_centers.append((lo + hi) / 2)
            pass_rates.append(passed / len(in_bin) * 100)
            bin_counts.append(len(in_bin))
        ax.bar(bin_centers, pass_rates, width=8, color="#3498db", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Plate Width (px)")
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title("Pass Rate by Plate Width (10px bins, OCR-ran, min 3 samples)")
        for x, rate, cnt in zip(bin_centers, pass_rates, bin_counts):
            ax.text(x, rate + 1, f"n={cnt}", ha="center", va="bottom", fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "pass_rate_by_plate_width.png"), dpi=150)
        plt.close(fig)

    # 5. Car height vs plate width scatter
    print("  Plotting car_vs_plate_scatter...")
    scatter_rows = [r for r in all_flat if r["plate_width"] > 0]
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(10, 8))
        for rt in RESULT_TYPES:
            subset = [r for r in scatter_rows if r["result"] == rt]
            if not subset:
                continue
            ax.scatter(
                [r["car_height"] for r in subset],
                [r["plate_width"] for r in subset],
                c=color_map.get(rt, "#999"), label=rt,
                alpha=0.5, s=10, edgecolors="none",
            )
        ax.set_xlabel("Car Height (px)")
        ax.set_ylabel("Plate Width (px)")
        ax.set_title("Car Height vs Plate Width (color = result)")
        ax.legend(fontsize=7, loc="upper left", markerscale=3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "car_vs_plate_scatter.png"), dpi=150)
        plt.close(fig)

    # 6. OCR confidence vs plate width
    print("  Plotting ocr_conf_vs_plate_width...")
    conf_rows = [r for r in all_flat if r["ocr_ran"] and r["plate_width"] > 0 and r["ocr_conf"] > 0]
    if conf_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ecc71" if r["result"] == "passed" else "#e74c3c" for r in conf_rows]
        ax.scatter(
            [r["plate_width"] for r in conf_rows],
            [r["ocr_conf"] for r in conf_rows],
            c=colors, alpha=0.5, s=10, edgecolors="none",
        )
        ax.set_xlabel("Plate Width (px)")
        ax.set_ylabel("OCR Confidence")
        ax.set_title("OCR Confidence vs Plate Width (green=passed, red=failed)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "ocr_conf_vs_plate_width.png"), dpi=150)
        plt.close(fig)

    # 7. Per-video rates (bar chart)
    print("  Plotting per_video_rates...")
    fig, ax = plt.subplots(figsize=(12, 6))
    video_names = list(all_rows.keys())
    ocr_rates = []
    pass_rates = []
    for vn in video_names:
        rows = all_rows[vn]
        total = len(rows)
        ocr = sum(1 for r in rows if r["ocr_ran"])
        passed = sum(1 for r in rows if r["result"] == "passed")
        ocr_rates.append(ocr / total * 100 if total else 0)
        pass_rates.append(passed / total * 100 if total else 0)

    x = np.arange(len(video_names))
    width = 0.35
    ax.bar(x - width / 2, ocr_rates, width, label="OCR Ran %", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, pass_rates, width, label="Passed %", color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Video")
    ax.set_ylabel("Rate (%)")
    ax.set_title("OCR Run Rate and Pass Rate per Video")
    ax.set_xticks(x)
    ax.set_xticklabels(video_names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_video_rates.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


def main():
    parser = argparse.ArgumentParser(description="OCR Experiment Tool")
    parser.add_argument("--videos", nargs="+", help="Specific video paths to process")
    args = parser.parse_args()

    # Output directory
    out_dir = os.path.join(BASE_DIR, "ocr_experiments")
    os.makedirs(out_dir, exist_ok=True)

    # Load config
    cfg = load_yaml(CONFIG_PATH)

    # Determine videos
    if args.videos:
        video_paths = args.videos
    else:
        video_paths = detect_video_files()

    if not video_paths:
        print("No video files found. Use --videos to specify paths.")
        return

    print(f"Videos to process: {len(video_paths)}")
    for vp in video_paths:
        print(f"  - {vp}")

    # Load models once
    yolo, recognizer, use_half, device = load_models(cfg)

    # Process each video
    all_rows = {}
    t_total = time.time()

    for video_path in video_paths:
        video_name = Path(video_path).stem
        rows = process_video(video_path, yolo, recognizer, use_half, device, cfg, out_dir)
        all_rows[video_name] = rows

    total_elapsed = time.time() - t_total
    total_dets = sum(len(rows) for rows in all_rows.values())
    print(f"\n{'='*60}")
    print(f"ALL DONE: {len(video_paths)} videos, {total_dets} detections, "
          f"{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Generate outputs
    generate_summary(all_rows, out_dir)
    generate_report(all_rows, out_dir)
    generate_plots(all_rows, out_dir)


if __name__ == "__main__":
    main()
