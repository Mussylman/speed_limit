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
    corr = args.correction

    # Видео-запись
    video_name = Path(args.video).stem
    out_dir = ROOT / "outputs" / f"distance_{video_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "distance.mp4")
    tmp_path = str(out_dir / "distance_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))
    print(f"Видео -> {out_path}\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, tracker="botsort.yaml",
            classes=[2, 5, 7],  # car, bus, truck
            verbose=False, imgsz=args.imgsz,
        )

        boxes = results[0].boxes
        if boxes.id is not None:
            ids = boxes.id.int().cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

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

    for tid in sorted(tracks.keys()):
        pts = tracks[tid]
        if len(pts) < 3:
            continue

        # расстояние от камеры в начале и конце
        cam_xc = cam_x * corr
        cam_yc = cam_y * corr
        d_start = np.sqrt((pts[0][1] - cam_xc) ** 2 + (pts[0][2] - cam_yc) ** 2)
        d_end = np.sqrt((pts[-1][1] - cam_xc) ** 2 + (pts[-1][2] - cam_yc) ** 2)

        # путь
        total = 0.0
        for j in range(1, len(pts)):
            dx = pts[j][1] - pts[j - 1][1]
            dy = pts[j][2] - pts[j - 1][2]
            total += np.sqrt(dx * dx + dy * dy)

        dur = (pts[-1][0] - pts[0][0]) / fps
        print(f"{tid:<6}{d_start:<12.1f}{d_end:<12.1f}{total:<12.1f}{dur:<10.1f}")

    print()


if __name__ == "__main__":
    main()
