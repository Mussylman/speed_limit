# calibrate_homography.py
# Утилита для калибровки гомографии (перспектива → bird's eye view)
# Кликаете 4 точки на дороге, вводите реальные размеры → получаете матрицу

import cv2
import numpy as np
import yaml
import argparse
import os

points = []
frame_display = None


def click_event(event, x, y, flags, param):
    """Обработчик кликов мыши"""
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"📍 Точка {len(points)}: ({x}, {y})")

        # Рисуем точку
        cv2.circle(frame_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_display, str(len(points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Рисуем линии между точками
        if len(points) > 1:
            cv2.line(frame_display, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(frame_display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
            print("\n✅ 4 точки выбраны! Нажмите любую клавишу для продолжения...")

        cv2.imshow("Calibration", frame_display)


def main():
    global frame_display, points

    parser = argparse.ArgumentParser(description="Калибровка гомографии для измерения скорости")
    parser.add_argument("--video", required=True, help="Путь к видео или изображению")
    parser.add_argument("--output", default="homography_config.yaml", help="Файл для сохранения")
    args = parser.parse_args()

    # Загружаем кадр
    if args.video.lower().endswith(('.jpg', '.png', '.jpeg')):
        frame = cv2.imread(args.video)
    else:
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("❌ Не удалось прочитать видео")
            return

    frame_display = frame.copy()
    h, w = frame.shape[:2]

    print("=" * 60)
    print("🎯 КАЛИБРОВКА ГОМОГРАФИИ")
    print("=" * 60)
    print(f"Разрешение: {w}x{h}")
    print()
    print("📌 Инструкция:")
    print("   Кликните 4 точки на дороге, образующие прямоугольник")
    print("   в РЕАЛЬНОМ мире (например, углы полосы движения)")
    print()
    print("   Порядок точек:")
    print("   1 ●─────────● 2    (дальняя сторона)")
    print("     │         │")
    print("     │ дорога  │")
    print("     │         │")
    print("   4 ●─────────● 3    (ближняя сторона)")
    print()
    print("   Лучше всего использовать разметку или края полосы")
    print("=" * 60)

    cv2.imshow("Calibration", frame_display)
    cv2.setMouseCallback("Calibration", click_event)

    # Ждём 4 клика
    while len(points) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            print("❌ Отменено")
            cv2.destroyAllWindows()
            return

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Запрашиваем реальные размеры
    print()
    print("📐 Введите реальные размеры прямоугольника:")

    try:
        width_m = float(input("   Ширина (точки 1-2 и 4-3) в метрах [3.5]: ") or "3.5")
        length_m = float(input("   Длина (точки 1-4 и 2-3) в метрах [20]: ") or "20")
    except ValueError:
        print("❌ Неверный ввод")
        return

    # Исходные точки (пиксели)
    src_pts = np.float32(points)

    # Целевые точки (метры, bird's eye view)
    # Масштаб: 100 пикселей = 1 метр (для визуализации)
    scale = 100
    dst_pts = np.float32([
        [0, 0],
        [width_m * scale, 0],
        [width_m * scale, length_m * scale],
        [0, length_m * scale]
    ])

    # Вычисляем матрицу гомографии
    H, status = cv2.findHomography(src_pts, dst_pts)

    # Сохраняем
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

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True, default_flow_style=None)

    # Также сохраняем матрицу отдельно
    matrix_file = args.output.replace(".yaml", "_matrix.npy")
    np.save(matrix_file, H)

    print()
    print("=" * 60)
    print("✅ КАЛИБРОВКА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"📄 Конфиг: {args.output}")
    print(f"📄 Матрица: {matrix_file}")
    print()
    print("Исходные точки (пиксели):")
    for i, p in enumerate(points, 1):
        print(f"   {i}: {p}")
    print()
    print(f"Реальные размеры: {width_m}м × {length_m}м")
    print()
    print("Матрица гомографии:")
    print(H)
    print()

    # Показываем bird's eye view для проверки
    print("🔍 Показываю Bird's Eye View для проверки...")

    bev_w = int(width_m * scale) + 100
    bev_h = int(length_m * scale) + 100

    # Смещаем целевые точки для отступа
    dst_pts_shifted = dst_pts + np.array([50, 50])
    H_display, _ = cv2.findHomography(src_pts, dst_pts_shifted)

    bev = cv2.warpPerspective(frame, H_display, (bev_w, bev_h))

    # Рисуем сетку (каждый метр)
    for i in range(int(width_m) + 1):
        x = 50 + i * scale
        cv2.line(bev, (x, 50), (x, 50 + int(length_m * scale)), (0, 255, 0), 1)
    for i in range(int(length_m) + 1):
        y = 50 + i * scale
        cv2.line(bev, (50, y), (50 + int(width_m * scale), y), (0, 255, 0), 1)

    cv2.putText(bev, "Bird's Eye View (1 клетка = 1м)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Bird's Eye View", bev)
    cv2.imshow("Original", frame)
    print("Нажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
