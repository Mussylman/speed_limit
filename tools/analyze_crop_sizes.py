"""Analyze OCR accuracy vs car crop size from exp_59 debug data.

Combines car_crops (with orig_WIDTHxHEIGHT in filename) and plate_bbox
(with OCR text in filename) to determine at what crop sizes NomeroffNet
reads plates correctly per camera.

Usage:
    python tools/analyze_crop_sizes.py outputs/exp_59
"""
import sys
import os
import re
import glob
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Known correct plates for verification
CORRECT_PLATES = {
    "791AL01", "976ABR01", "975ZCK01", "413AZN13", "383BAW01"
}


def parse_car_crops(car_crops_dir):
    """Parse car_crop filenames to get (seq, frame, track_id, width, height)."""
    crops = {}
    for f in sorted(glob.glob(os.path.join(car_crops_dir, "*.jpg"))):
        fname = os.path.basename(f)
        # Pattern: 00001_f419_t184_orig_350x317.jpg
        m = re.match(r"(\d+)_f(\d+)_t(\d+)_orig_(\d+)x(\d+)\.jpg", fname)
        if m:
            seq, frame, tid, w, h = int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])
            crops[(frame, tid)] = {"seq": seq, "width": w, "height": h, "area": w * h,
                                   "aspect": round(h / w, 2) if w > 0 else 0}
    return crops


def parse_plate_bbox(plate_bbox_dir):
    """Parse plate_bbox filenames to get (seq, frame, track_id, scale, text)."""
    results = {}
    for f in sorted(glob.glob(os.path.join(plate_bbox_dir, "*.jpg"))):
        fname = os.path.basename(f)
        # Pattern: 00009_f419_t184_s1.0_791AL01.jpg
        m = re.match(r"(\d+)_f(\d+)_t(\d+)_s([\d.]+)_(.+)\.jpg", fname)
        if m:
            seq, frame, tid, scale, text = int(m[1]), int(m[2]), int(m[3]), m[4], m[5]
            key = (frame, tid)
            if key not in results:
                results[key] = []
            results[key].append({"seq": seq, "scale": scale, "text": text})
    return results


def analyze_camera(cam_dir, cam_name):
    """Analyze one camera's data."""
    car_crops_dir = os.path.join(cam_dir, "debug_ocr", "car_crops")
    plate_bbox_dir = os.path.join(cam_dir, "debug_ocr", "plate_bbox")

    if not os.path.isdir(car_crops_dir):
        print(f"  No car_crops dir: {car_crops_dir}")
        return
    if not os.path.isdir(plate_bbox_dir):
        print(f"  No plate_bbox dir: {plate_bbox_dir}")
        return

    crops = parse_car_crops(car_crops_dir)
    ocr_results = parse_plate_bbox(plate_bbox_dir)

    if not crops:
        print(f"  No car_crops found")
        return

    # Group by track_id
    tracks = defaultdict(list)
    for (frame, tid), info in sorted(crops.items()):
        tracks[tid].append((frame, info))

    print(f"\n{'=' * 100}")
    print(f"  {cam_name}: {len(crops)} car_crops, {len(ocr_results)} OCR frames, {len(tracks)} tracks")
    print(f"{'=' * 100}")

    # For each track, show crop sizes and OCR results
    for tid in sorted(tracks.keys()):
        frames = tracks[tid]
        widths = [info["width"] for _, info in frames]
        heights = [info["height"] for _, info in frames]
        areas = [info["area"] for _, info in frames]

        # Get OCR results for this track
        track_ocr = []
        for (frame, t), ocr_list in ocr_results.items():
            if t == tid:
                for ocr in ocr_list:
                    # Find matching car_crop size for this frame
                    crop_info = crops.get((frame, tid), None)
                    track_ocr.append({
                        "frame": frame,
                        "scale": ocr["scale"],
                        "text": ocr["text"],
                        "crop_w": crop_info["width"] if crop_info else 0,
                        "crop_h": crop_info["height"] if crop_info else 0,
                        "crop_area": crop_info["area"] if crop_info else 0,
                    })

        # Determine correct plate for this track
        texts = [o["text"] for o in track_ocr]
        correct_text = None
        for t in texts:
            if t in CORRECT_PLATES:
                correct_text = t
                break

        print(f"\n  Track t{tid}: {len(frames)} crops, {len(track_ocr)} OCR reads")
        print(f"    Crop width  range: {min(widths):4d} - {max(widths):4d} px")
        print(f"    Crop height range: {min(heights):4d} - {max(heights):4d} px")
        print(f"    Crop area   range: {min(areas):6d} - {max(areas):6d} px²")
        if correct_text:
            print(f"    Expected plate: {correct_text}")
        print()

        # Show each OCR read with crop size
        print(f"    {'frame':>6} {'scale':>5} {'crop_w':>6} {'crop_h':>6} {'area':>8} {'text':<15} {'correct':>7}")
        print(f"    {'-'*60}")
        for o in sorted(track_ocr, key=lambda x: (x["frame"], x["scale"])):
            is_correct = "✓" if correct_text and o["text"] == correct_text else "✗" if correct_text else "?"
            print(f"    {o['frame']:6d} {o['scale']:>5} {o['crop_w']:6d} {o['crop_h']:6d} {o['crop_area']:8d} {o['text']:<15} {is_correct:>7}")

        # Size stats for correct vs incorrect reads
        if correct_text and len(track_ocr) > 1:
            correct_reads = [o for o in track_ocr if o["text"] == correct_text]
            wrong_reads = [o for o in track_ocr if o["text"] != correct_text]
            print(f"\n    CORRECT reads ({len(correct_reads)}):")
            if correct_reads:
                cw = [o["crop_w"] for o in correct_reads]
                ch = [o["crop_h"] for o in correct_reads]
                ca = [o["crop_area"] for o in correct_reads]
                print(f"      Width:  {min(cw):4d} - {max(cw):4d}  avg={sum(cw)/len(cw):.0f}")
                print(f"      Height: {min(ch):4d} - {max(ch):4d}  avg={sum(ch)/len(ch):.0f}")
                print(f"      Area:   {min(ca):6d} - {max(ca):6d}  avg={sum(ca)/len(ca):.0f}")
            print(f"    WRONG reads ({len(wrong_reads)}):")
            if wrong_reads:
                cw = [o["crop_w"] for o in wrong_reads]
                ch = [o["crop_h"] for o in wrong_reads]
                ca = [o["crop_area"] for o in wrong_reads]
                print(f"      Width:  {min(cw):4d} - {max(cw):4d}  avg={sum(cw)/len(cw):.0f}")
                print(f"      Height: {min(ch):4d} - {max(ch):4d}  avg={sum(ch)/len(ch):.0f}")
                print(f"      Area:   {min(ca):6d} - {max(ca):6d}  avg={sum(ca)/len(ca):.0f}")
                unique_wrong = set(o["text"] for o in wrong_reads)
                print(f"      Texts:  {', '.join(sorted(unique_wrong))}")

    # Overall summary: what crop sizes give correct results?
    print(f"\n{'=' * 100}")
    print(f"  {cam_name} — OVERALL SIZE ANALYSIS")
    print(f"{'=' * 100}")

    all_correct = []
    all_wrong = []
    all_no_plate = []

    for tid in sorted(tracks.keys()):
        for (frame, t), ocr_list in ocr_results.items():
            if t == tid:
                crop_info = crops.get((frame, tid), None)
                if not crop_info:
                    continue
                for ocr in ocr_list:
                    entry = {"w": crop_info["width"], "h": crop_info["height"],
                             "area": crop_info["area"], "text": ocr["text"]}
                    if ocr["text"] in CORRECT_PLATES:
                        all_correct.append(entry)
                    else:
                        all_wrong.append(entry)

    # Count no-plate frames (crops without any OCR result)
    for (frame, tid), info in crops.items():
        if (frame, tid) not in ocr_results:
            all_no_plate.append({"w": info["width"], "h": info["height"], "area": info["area"]})

    print(f"\n  Correct OCR ({len(all_correct)} reads):")
    if all_correct:
        ws = [e["w"] for e in all_correct]
        hs = [e["h"] for e in all_correct]
        ars = [e["area"] for e in all_correct]
        print(f"    Width:  {min(ws):4d} - {max(ws):4d}  avg={sum(ws)/len(ws):.0f}")
        print(f"    Height: {min(hs):4d} - {max(hs):4d}  avg={sum(hs)/len(hs):.0f}")
        print(f"    Area:   {min(ars):6d} - {max(ars):6d}  avg={sum(ars)/len(ars):.0f}")

    print(f"\n  Wrong OCR ({len(all_wrong)} reads):")
    if all_wrong:
        ws = [e["w"] for e in all_wrong]
        hs = [e["h"] for e in all_wrong]
        ars = [e["area"] for e in all_wrong]
        print(f"    Width:  {min(ws):4d} - {max(ws):4d}  avg={sum(ws)/len(ws):.0f}")
        print(f"    Height: {min(hs):4d} - {max(hs):4d}  avg={sum(hs)/len(hs):.0f}")
        print(f"    Area:   {min(ars):6d} - {max(ars):6d}  avg={sum(ars)/len(ars):.0f}")
        unique = set(e["text"] for e in all_wrong)
        print(f"    Texts:  {', '.join(sorted(unique))}")

    print(f"\n  No plate detected ({len(all_no_plate)} crops — no OCR attempted or no plate found):")
    if all_no_plate:
        ws = [e["w"] for e in all_no_plate]
        hs = [e["h"] for e in all_no_plate]
        print(f"    Width:  {min(ws):4d} - {max(ws):4d}  avg={sum(ws)/len(ws):.0f}")
        print(f"    Height: {min(hs):4d} - {max(hs):4d}  avg={sum(hs)/len(hs):.0f}")

    # Width bucket analysis
    print(f"\n  Width bucket analysis (correct / total OCR reads):")
    buckets = [(0, 300), (300, 350), (350, 400), (400, 450), (450, 500), (500, 550), (550, 600), (600, 700), (700, 1000)]
    for lo, hi in buckets:
        c = sum(1 for e in all_correct if lo <= e["w"] < hi)
        w = sum(1 for e in all_wrong if lo <= e["w"] < hi)
        total = c + w
        if total > 0:
            pct = c / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {lo:4d}-{hi:4d}px: {c:3d}/{total:3d} ({pct:5.1f}%) {bar}")

    # Area bucket analysis
    print(f"\n  Area bucket analysis (correct / total OCR reads):")
    area_buckets = [(0, 100000), (100000, 120000), (120000, 150000), (150000, 180000),
                    (180000, 220000), (220000, 280000), (280000, 400000)]
    for lo, hi in area_buckets:
        c = sum(1 for e in all_correct if lo <= e["area"] < hi)
        w = sum(1 for e in all_wrong if lo <= e["area"] < hi)
        total = c + w
        if total > 0:
            pct = c / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {lo:6d}-{hi:6d}px²: {c:3d}/{total:3d} ({pct:5.1f}%) {bar}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_crop_sizes.py <exp_output_dir>")
        print("Example: python tools/analyze_crop_sizes.py outputs/exp_59")
        sys.exit(1)

    exp_dir = sys.argv[1]
    if not os.path.isdir(exp_dir):
        print(f"Not a directory: {exp_dir}")
        sys.exit(1)

    # Find camera directories
    for entry in sorted(os.listdir(exp_dir)):
        cam_dir = os.path.join(exp_dir, entry)
        if os.path.isdir(cam_dir) and entry.startswith("Camera_"):
            print(f"\nProcessing {entry}...")
            analyze_camera(cam_dir, entry)


if __name__ == "__main__":
    main()
