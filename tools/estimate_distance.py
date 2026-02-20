"""
estimate_distance.py
Оценка расстояния до автомобиля, движущегося от камеры.

Берёт гомографию из конфига, трансформирует нижнюю точку bbox
в реальные метры и показывает:
  - расстояние от камеры (м)
  - пройденный путь (м)

Использование:
    python tools/estimate_distance.py videos/test_8.mp4
    python tools/estimate_distance.py videos/test_8.mp4 --no-show
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from collections import defaultdict

# Добавляем корень проекта в path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_homography(config_path: str):
    """Загрузить матрицу гомографии и параметры из yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hom = cfg.get("homography", cfg)
    H = np.array(hom["matrix"], dtype=np.float64)
    scale = hom.get("scale_px_per_m", 100)
    length_m = hom.get("real_length_m", 60.0)
    return H, scale, length_m


def px_to_meters(H, scale, px, py):
    """Пиксели камеры -> метры BEV."""
    pt = np.array([px, py, 1.0], dtype=np.float64)
    t = H @ pt
    if abs(t[2]) < 1e-10:
        return None, None
    return (t[0] / t[2]) / scale, (t[1] / t[2]) / scale


def smooth_path_length(pts, window=15):
    """Сглаженный путь: скользящее среднее по BEV координатам, затем сумма сегментов."""
    if len(pts) < 3:
        return 0.0
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    if len(xs) < window:
        window = max(3, len(xs) // 2 * 2 + 1)
    else:
        window = window if window % 2 == 1 else window + 1
    # Moving average
    kernel = np.ones(window) / window
    xs_s = np.convolve(xs, kernel, mode='valid')
    ys_s = np.convolve(ys, kernel, mode='valid')
    # Accumulated path on smoothed
    total = 0.0
    for i in range(1, len(xs_s)):
        total += np.sqrt((xs_s[i] - xs_s[i-1])**2 + (ys_s[i] - ys_s[i-1])**2)
    return total


def main():
    ap = argparse.ArgumentParser(description="Оценка расстояния до авто")
    ap.add_argument("video", help="Путь к видео")
    ap.add_argument("--homography", default=str(ROOT / "config" / "homography_config.yaml"),
                    help="Путь к YAML гомографии")
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo11n.pt"),
                    help="Путь к YOLO модели")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--no-show", action="store_true", help="Без окна")
    ap.add_argument("--correction", type=float, default=0.43,
                    help="Коррекция расстояния (из speed_correction)")
    args = ap.parse_args()

    # ---------- загрузка ----------
    from ultralytics import YOLO
    model = YOLO(args.model)
    H, scale, length_m = load_homography(args.homography)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Не удалось открыть: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Опорная точка камеры — нижний центр кадра (ближайшая к камере)
    cam_x, cam_y = px_to_meters(H, scale, frame_w / 2, frame_h)
    print(f"Видео: {args.video}  ({frame_w}x{frame_h} @ {fps:.0f} fps)")
    print(f"Камера BEV: ({cam_x:.1f}, {cam_y:.1f}) м")
    print(f"Коррекция: {args.correction}")
    print(f"[q] — выход\n")

    # Треки: {track_id: [(frame, x_m, y_m), ...]}
    tracks: dict = defaultdict(list)
    # Лучший кроп для каждого трека (наибольший bbox)
    best_crops: dict = {}  # {track_id: (area, crop_img)}
    corr = args.correction

    # Видео-запись
    video_path = Path(args.video)
    # Если видео называется fixed.mp4, берём имя родительской папки
    if video_path.stem == "fixed" or video_path.stem == "Camera_12_fixed":
        video_name = video_path.parent.name
    else:
        video_name = video_path.stem
    out_dir = ROOT / "outputs" / f"{video_name}_exp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "distance.mp4")
    tmp_path = str(out_dir / "distance_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))
    print(f"Видео -> {out_path}\n")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use track for long videos, predict for short (1-2 frames)
        if total_frames > 5:
            results = model.track(
                frame, persist=True, tracker="botsort.yaml",
                classes=[2, 5, 7],  # car, bus, truck
                verbose=False, imgsz=args.imgsz,
            )
        else:
            results = model.predict(
                frame, classes=[2, 5, 7],
                verbose=False, imgsz=args.imgsz,
            )

        boxes = results[0].boxes
        has_ids = boxes.id is not None
        if has_ids:
            ids = boxes.id.int().cpu().numpy()
        else:
            ids = list(range(len(boxes)))
        xyxy = boxes.xyxy.cpu().numpy()

        if len(xyxy) > 0:
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                # Нижний центр bbox (точка контакта с дорогой)
                bc_x = (x1 + x2) / 2
                bc_y = y2

                xm, ym = px_to_meters(H, scale, bc_x, bc_y)
                if xm is None:
                    continue

                # Применяем коррекцию
                xm *= corr
                ym *= corr

                cam_xc = cam_x * corr
                cam_yc = cam_y * corr

                tracks[tid].append((frame_idx, xm, ym))

                # Сохраняем лучший кроп (по площади bbox)
                area = (x2 - x1) * (y2 - y1)
                if tid not in best_crops or area > best_crops[tid][0]:
                    pad = 20
                    cy1 = max(0, int(y1) - pad)
                    cy2 = min(frame_h, int(y2) + pad)
                    cx1 = max(0, int(x1) - pad)
                    cx2 = min(frame_w, int(x2) + pad)
                    best_crops[tid] = (area, frame[cy1:cy2, cx1:cx2].copy())

                # Расстояние от камеры (евклидово в BEV)
                dist = np.sqrt((xm - cam_xc) ** 2 + (ym - cam_yc) ** 2)

                # Пройденный путь (сумма сегментов)
                pts = tracks[tid]
                traveled = 0.0
                for j in range(1, len(pts)):
                    dx = pts[j][1] - pts[j - 1][1]
                    dy = pts[j][2] - pts[j - 1][2]
                    traveled += np.sqrt(dx * dx + dy * dy)

                # --- рисуем ---
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Расстояние
                label1 = f"ID:{tid}  {dist:.1f} m"
                cv2.putText(frame, label1, (int(x1), int(y1) - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Пройдено
                label2 = f"path: {traveled:.1f} m"
                cv2.putText(frame, label2, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Инфо-панель
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)

        if not args.no_show:
            cv2.imshow("Distance Estimation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Перекодируем в H.264 через ffmpeg (совместимо с WhatsApp)
    import subprocess
    ret_ff = subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ], capture_output=True)
    if ret_ff.returncode == 0:
        Path(tmp_path).unlink(missing_ok=True)
        print(f"\nВидео сохранено (H.264): {out_path}")
    else:
        # ffmpeg не найден — оставляем mp4v
        Path(tmp_path).rename(out_path)
        print(f"\nВидео сохранено (mp4v): {out_path}")

    # ========== итог ==========
    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"{'ID':<6}{'Начало,м':<12}{'Конец,м':<12}{'Путь,м':<12}{'Время,с':<10}")
    print("-" * 60)

    cam_xc = cam_x * corr
    cam_yc = cam_y * corr
    for tid in sorted(tracks.keys()):
        pts = tracks[tid]
        if len(pts) < 1:
            continue

        d_start = np.sqrt((pts[0][1] - cam_xc) ** 2 + (pts[0][2] - cam_yc) ** 2)
        d_end = np.sqrt((pts[-1][1] - cam_xc) ** 2 + (pts[-1][2] - cam_yc) ** 2)
        path = smooth_path_length(pts) if len(pts) >= 3 else 0.0
        dur = (pts[-1][0] - pts[0][0]) / fps if len(pts) > 1 else 0.0
        print(f"{tid:<6}{d_start:<12.1f}{d_end:<12.1f}{path:<12.1f}{dur:<10.1f}")

    print()

    # ========== карточки машин ==========
    from PIL import Image, ImageDraw, ImageFont
    cards_dir = out_dir / "vehicles"
    if cards_dir.exists():
        for old in cards_dir.glob("*.jpg"):
            old.unlink()
    cards_dir.mkdir(exist_ok=True)

    try:
        font_id = ImageFont.truetype("arialbd.ttf", 28)
        font_val = ImageFont.truetype("arial.ttf", 22)
        font_label = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font_id = font_val = font_label = ImageFont.load_default()

    cam_xc = cam_x * corr
    cam_yc = cam_y * corr
    card_idx = 0

    for tid in sorted(tracks.keys()):
        pts = tracks[tid]
        if len(pts) < 3:
            continue
        if tid not in best_crops:
            continue

        d_start = np.sqrt((pts[0][1] - cam_xc) ** 2 + (pts[0][2] - cam_yc) ** 2)
        d_end = np.sqrt((pts[-1][1] - cam_xc) ** 2 + (pts[-1][2] - cam_yc) ** 2)
        path = smooth_path_length(pts)
        dur = (pts[-1][0] - pts[0][0]) / fps

        _, crop_bgr = best_crops[tid]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # Scale crop to fixed height
        CROP_H = 300
        scale = CROP_H / crop_pil.height
        crop_pil = crop_pil.resize(
            (int(crop_pil.width * scale), CROP_H), Image.LANCZOS
        )

        CARD_W = max(crop_pil.width + 40, 420)
        INFO_H = 160
        CARD_H = CROP_H + INFO_H + 20

        card = Image.new("RGB", (CARD_W, CARD_H), (30, 30, 35))
        draw = ImageDraw.Draw(card)

        # Paste crop centered
        cx = (CARD_W - crop_pil.width) // 2
        card.paste(crop_pil, (cx, 10))

        # Info panel
        iy = CROP_H + 20
        draw.text((20, iy), f"ID: {tid}", font=font_id, fill=(255, 255, 255))

        iy += 38
        labels = [
            ("Начало:", f"{d_start:.1f} м"),
            ("Конец:", f"{d_end:.1f} м"),
            ("Путь:", f"{path:.1f} м"),
            ("Время:", f"{dur:.1f} с"),
        ]
        for lbl, val in labels:
            draw.text((20, iy), lbl, font=font_label, fill=(170, 170, 180))
            draw.text((100, iy), val, font=font_val, fill=(80, 220, 100))
            iy += 28

        card_path = cards_dir / f"{card_idx:03d}_id{tid}_{path:.1f}m.jpg"
        card.save(str(card_path), quality=95)
        card_idx += 1

    print(f"Карточки: {cards_dir}  ({card_idx} шт.)")


if __name__ == "__main__":
    main()
