# report_generator.py
# Generates a client-facing report/ folder with vehicle cards,
# violations, summary image, text report, and video copy.

import os
import shutil
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


class ReportGenerator:
    """Builds a presentable report/ directory from pipeline results."""

    CARD_W = 700
    CROP_W = 660
    THUMB_SIZE = 160
    THUMBS_PER_ROW = 7
    SUMMARY_W = 1280

    # Colours (BGR)
    BG_DARK = (30, 30, 30)
    BG_VIOLATION = (25, 15, 40)
    BORDER_NORMAL = (120, 120, 120)
    BORDER_VIOLATION = (0, 0, 220)
    GREEN = (0, 200, 0)
    RED = (0, 0, 230)
    WHITE = (255, 255, 255)
    GRAY = (160, 160, 160)
    YELLOW = (0, 220, 255)

    def __init__(self, output_dir: str, camera_id: str, speed_limit: int,
                 address: str = "", coordinates: tuple = None):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.speed_limit = speed_limit
        self.address = address
        self.coordinates = coordinates  # (lat, lon) or None
        self.report_dir = os.path.join(output_dir, "report")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def generate(self, passed_results, speed_tracker, video_path,
                 start_time: datetime.datetime, duration_sec: float):
        """Main entry point — creates the full report/ directory."""
        os.makedirs(os.path.join(self.report_dir, "vehicles"), exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "violations"), exist_ok=True)

        vehicles = self._merge_data(passed_results, speed_tracker)
        vehicles.sort(key=lambda v: v["timestamp"])

        stats = self._compute_stats(vehicles)

        # Vehicle cards
        for idx, v in enumerate(vehicles, 1):
            v["idx"] = idx
            card = self._create_vehicle_card(v, idx)
            fname = self._vehicle_filename(v, idx)
            v["filename"] = fname
            path = os.path.join(self.report_dir, "vehicles", fname)
            cv2.imwrite(path, card)
            if v["is_violation"]:
                shutil.copy2(path,
                             os.path.join(self.report_dir, "violations", fname))

        # Summary image
        summary = self._create_summary(vehicles, stats, start_time, duration_sec)
        cv2.imwrite(os.path.join(self.report_dir, "summary.jpg"), summary)

        # Text report
        self._write_text_report(vehicles, stats, start_time, duration_sec)

        # Copy video
        if video_path and os.path.isfile(video_path):
            shutil.copy2(video_path,
                         os.path.join(self.report_dir, "video.mp4"))

        total = len(vehicles)
        viols = stats["violations"]
        print(f"Report: {self.report_dir}  ({total} vehicles, {viols} violations)")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _merge_data(self, passed_results, speed_tracker):
        """Join speed_tracker.speeds with passed_results by track_id."""
        vehicles = []
        for track_id, se in speed_tracker.speeds.items():
            pe = passed_results.get(track_id)
            plate_text = pe.plate_text if pe else "UNKNOWN"
            crop = pe.crop if pe and pe.crop is not None else None
            car_score = pe.car_score if pe else 0.0
            plate_score = pe.plate_score if pe else 0.0
            ocr_score = pe.ocr_score if pe else 0.0

            # Prefer OCR timestamp (when photo was taken) over speed timestamp
            ts = pe.timestamp if pe and pe.timestamp else se.timestamp

            vehicles.append({
                "track_id": track_id,
                "plate_text": plate_text,
                "speed_kmh": se.speed_kmh,
                "is_violation": bool(se.speed_kmh > self.speed_limit),
                "timestamp": ts,
                "crop": crop,
                "car_score": car_score,
                "plate_score": plate_score,
                "ocr_score": ocr_score,
                "camera_label": pe.camera_label if pe and hasattr(pe, "camera_label") else "",
            })
        return vehicles

    def _compute_stats(self, vehicles):
        total = len(vehicles)
        with_plate = sum(1 for v in vehicles if v["plate_text"] != "UNKNOWN")
        violations = sum(1 for v in vehicles if v["is_violation"])
        speeds = [v["speed_kmh"] for v in vehicles if v["speed_kmh"] > 0]
        max_speed = max(speeds) if speeds else 0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        return {
            "total": total,
            "with_plate": with_plate,
            "violations": violations,
            "max_speed": max_speed,
            "avg_speed": avg_speed,
        }

    @staticmethod
    def _vehicle_filename(v, idx):
        plate = v["plate_text"].replace(" ", "")
        spd = int(v["speed_kmh"])
        tag = "_VIOLATION" if v["is_violation"] else ""
        return f"{idx:03d}_{plate}_{spd}kmh{tag}.jpg"

    # ------------------------------------------------------------------
    # Vehicle card
    # ------------------------------------------------------------------
    @staticmethod
    def _put_text_pil(img: np.ndarray, text: str, pos: tuple,
                      font_size: int = 20, color: tuple = (255, 255, 255)):
        """Draw Unicode text on OpenCV image via PIL. Color is BGR."""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=rgb_color)
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np.copyto(img, result)

    @staticmethod
    def _format_datetime(ts_str: str) -> str:
        """Convert ISO timestamp to human-readable format: 13.02.2026  14:08"""
        try:
            dt = datetime.datetime.fromisoformat(ts_str)
            return dt.strftime("%Y-%m-%d  %H:%M:%S")
        except Exception:
            return ts_str[:16].replace("T", "  ")

    def _create_vehicle_card(self, v, idx):
        is_viol = v["is_violation"]
        bg = self.BG_VIOLATION if is_viol else self.BG_DARK
        border_color = self.BORDER_VIOLATION if is_viol else self.BORDER_NORMAL
        border_t = 3

        # Scale crop
        crop_img = self._scale_crop(v["crop"])
        crop_h = crop_img.shape[0] if crop_img is not None else 200

        # Layout heights
        header_h = 70  # two lines: date + address
        crop_pad = 15
        info_h = 110
        card_h = header_h + crop_pad + crop_h + crop_pad + info_h

        card = np.full((card_h, self.CARD_W, 3), bg, dtype=np.uint8)

        y = 0
        # --- Header (dark panel with date + address) ---
        cv2.rectangle(card, (0, 0), (self.CARD_W, header_h), (45, 45, 45), -1)

        # Line 1: date (left) + camera label (right)
        date_str = self._format_datetime(v["timestamp"])
        cv2.putText(card, date_str, (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 1)
        cam_label = v.get("camera_label", "")
        if cam_label:
            lbl_size = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(card, cam_label, (self.CARD_W - lbl_size[0] - 15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.YELLOW, 2)

        # Line 2: address + coordinates (PIL for Cyrillic support)
        addr_line = self.address or ""
        if self.coordinates:
            coord_str = f"({self.coordinates[0]:.6f}, {self.coordinates[1]:.6f})"
            addr_line = f"{addr_line}   {coord_str}" if addr_line else coord_str
        if addr_line:
            self._put_text_pil(card, addr_line, (15, 35),
                               font_size=16, color=self.GRAY)
        y = header_h + crop_pad

        # --- Crop ---
        if crop_img is not None:
            x_off = (self.CARD_W - crop_img.shape[1]) // 2
            card[y:y + crop_h, x_off:x_off + crop_img.shape[1]] = crop_img
        else:
            ph_w, ph_h = 300, crop_h
            x_off = (self.CARD_W - ph_w) // 2
            cv2.rectangle(card, (x_off, y), (x_off + ph_w, y + ph_h),
                          (60, 60, 60), -1)
            cv2.putText(card, "No image", (x_off + 80, y + ph_h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.GRAY, 2)
        y += crop_h + crop_pad

        # --- Divider ---
        cv2.line(card, (15, y), (self.CARD_W - 15, y), (80, 80, 80), 1)
        y += 10

        # --- Plate text (large, left) ---
        plate_display = v["plate_text"]
        # Format plate with spaces: 413AZN13 → 413 AZN 13
        if len(plate_display) == 8 and plate_display[:3].isdigit() and plate_display[6:].isdigit():
            plate_display = f"{plate_display[:3]} {plate_display[3:6]} {plate_display[6:]}"
        cv2.putText(card, plate_display, (20, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, self.WHITE, 3)

        # --- Speed (right) ---
        spd_text = f"{v['speed_kmh']:.0f} km/h"
        spd_color = self.RED if is_viol else self.GREEN
        spd_size = cv2.getTextSize(spd_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
        cv2.putText(card, spd_text, (self.CARD_W - spd_size[0] - 20, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, spd_color, 3)

        # Limit (under speed)
        limit_text = f"Limit: {self.speed_limit} km/h"
        lt_size = cv2.getTextSize(limit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(card, limit_text,
                    (self.CARD_W - lt_size[0] - 20, y + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.GRAY, 1)

        # Violation banner
        if is_viol:
            excess = v["speed_kmh"] - self.speed_limit
            banner_y = y + 75
            cv2.rectangle(card, (15, banner_y), (self.CARD_W - 15, banner_y + 30),
                          (0, 0, 180), -1)
            banner_text = f"VIOLATION  +{excess:.0f} km/h"
            bt_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            bx = (self.CARD_W - bt_size[0]) // 2
            cv2.putText(card, banner_text, (bx, banner_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)

        # Border
        cv2.rectangle(card, (0, 0), (self.CARD_W - 1, card_h - 1),
                      border_color, border_t)
        return card

    def _scale_crop(self, crop):
        """Scale crop to CROP_W keeping aspect ratio. Returns None if no crop."""
        if crop is None:
            return None
        h, w = crop.shape[:2]
        if w == 0 or h == 0:
            return None
        scale = self.CROP_W / w
        new_h = int(h * scale)
        return cv2.resize(crop, (self.CROP_W, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Thumbnail helper
    # ------------------------------------------------------------------
    def _make_thumbnail(self, crop, size=160):
        """Resize crop to square thumbnail preserving aspect ratio with letterbox."""
        if crop is None:
            thumb = np.full((size, size, 3), 60, dtype=np.uint8)
            cv2.putText(thumb, "?", (size // 2 - 12, size // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.GRAY, 2)
            return thumb
        h, w = crop.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        thumb = np.full((size, size, 3), 30, dtype=np.uint8)
        y_off = (size - nh) // 2
        x_off = (size - nw) // 2
        thumb[y_off:y_off + nh, x_off:x_off + nw] = resized
        return thumb

    # ------------------------------------------------------------------
    # Summary image
    # ------------------------------------------------------------------
    def _create_summary(self, vehicles, stats, start_time, duration_sec):
        # Layout
        header_h = 80
        stats_h = 100
        gap = 20
        ts = self.THUMB_SIZE

        n = len(vehicles)
        rows = max(1, (n + self.THUMBS_PER_ROW - 1) // self.THUMBS_PER_ROW)
        grid_h = rows * (ts + 10) + 10
        total_h = header_h + stats_h + gap + grid_h + gap

        img = np.full((total_h, self.SUMMARY_W, 3), self.BG_DARK, dtype=np.uint8)
        y = 0

        # --- Header ---
        cv2.rectangle(img, (0, 0), (self.SUMMARY_W, header_h), (45, 45, 45), -1)
        title = f"Speed Monitoring Report  |  {self.camera_id}"
        cv2.putText(img, title, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.YELLOW, 2)
        date_str = start_time.strftime("%Y-%m-%d %H:%M")
        dur_min = int(duration_sec // 60)
        dur_sec = int(duration_sec % 60)
        sub = f"{date_str}   Duration: {dur_min}m {dur_sec}s"
        cv2.putText(img, sub, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GRAY, 1)
        y = header_h + 10

        # --- Stats blocks (6 boxes) ---
        labels = ["Vehicles", "With Plate", "Violations",
                  "Max Speed", "Avg Speed", "Limit"]
        values = [
            str(stats["total"]),
            str(stats["with_plate"]),
            str(stats["violations"]),
            f"{stats['max_speed']:.0f}",
            f"{stats['avg_speed']:.0f}",
            str(self.speed_limit),
        ]
        box_w = 190
        box_h = 70
        margin = (self.SUMMARY_W - 6 * box_w) // 7
        for i in range(6):
            bx = margin + i * (box_w + margin)
            by = y
            # Violation count highlighted
            if labels[i] == "Violations" and stats["violations"] > 0:
                cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h),
                              (0, 0, 100), -1)
            else:
                cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h),
                              (50, 50, 50), -1)
            cv2.putText(img, labels[i], (bx + 10, by + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.GRAY, 1)
            cv2.putText(img, values[i], (bx + 10, by + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.WHITE, 2)
        y += box_h + gap

        # --- Thumbnail grid ---
        for i, v in enumerate(vehicles):
            row = i // self.THUMBS_PER_ROW
            col = i % self.THUMBS_PER_ROW
            tx = margin + col * (ts + 10)
            ty = y + row * (ts + 10)
            thumb = self._make_thumbnail(v["crop"], ts)
            # Fit within image bounds
            if ty + ts > img.shape[0] or tx + ts > img.shape[1]:
                continue
            img[ty:ty + ts, tx:tx + ts] = thumb
            # Red border for violations
            if v["is_violation"]:
                cv2.rectangle(img, (tx, ty), (tx + ts - 1, ty + ts - 1),
                              self.BORDER_VIOLATION, 2)
            # Label below thumbnail: speed
            label = f"{v['speed_kmh']:.0f}"
            color = self.RED if v["is_violation"] else self.GREEN
            cv2.putText(img, label, (tx + 2, ty + ts - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return img

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    def _write_text_report(self, vehicles, stats, start_time, duration_sec):
        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("  SPEED MONITORING REPORT")
        lines.append(sep)
        lines.append(f"Camera:       {self.camera_id}")
        lines.append(f"Date:         {start_time.strftime('%Y-%m-%d')}")
        lines.append(f"Time:         {start_time.strftime('%H:%M:%S')}")
        dur_min = int(duration_sec // 60)
        dur_sec = int(duration_sec % 60)
        lines.append(f"Duration:     {dur_min}m {dur_sec}s")
        lines.append(f"Speed limit:  {self.speed_limit} km/h")
        lines.append("")

        lines.append("--- SUMMARY ---")
        lines.append(f"Total vehicles:    {stats['total']}")
        lines.append(f"Plates recognized: {stats['with_plate']}")
        lines.append(f"Violations:        {stats['violations']}")
        lines.append(f"Max speed:         {stats['max_speed']:.0f} km/h")
        lines.append(f"Avg speed:         {stats['avg_speed']:.0f} km/h")
        lines.append("")

        # Violations table
        viols = [v for v in vehicles if v["is_violation"]]
        lines.append("--- VIOLATIONS ---")
        if viols:
            hdr = f"{'#':<6}{'Plate':<13}{'Speed':<11}{'Excess':<11}{'Time'}"
            lines.append(hdr)
            lines.append("-" * 60)
            for i, v in enumerate(viols, 1):
                excess = f"+{v['speed_kmh'] - self.speed_limit:.0f}"
                lines.append(
                    f"{i:<6}{v['plate_text']:<13}"
                    f"{v['speed_kmh']:<11.0f}{excess:<11}{v['timestamp']}"
                )
        else:
            lines.append("None")
        lines.append("")

        # All vehicles table
        lines.append("--- ALL VEHICLES ---")
        hdr = f"{'#':<6}{'Plate':<13}{'Speed':<11}{'Status':<13}{'Time'}"
        lines.append(hdr)
        lines.append("-" * 60)
        for v in vehicles:
            status = "VIOLATION" if v["is_violation"] else "OK"
            plate = v["plate_text"] if v["plate_text"] != "UNKNOWN" else "---"
            lines.append(
                f"{v['idx']:<6}{plate:<13}"
                f"{v['speed_kmh']:<11.0f}{status:<13}{v['timestamp']}"
            )
        lines.append("")
        lines.append(sep)

        path = os.path.join(self.report_dir, "report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
