import cv2
import yaml
import argparse
import os

points = []


def click_event(event, x, y, flags, param):
    """Обработчик кликов мышки"""
    global points, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"📍 Точка {len(points)}: {x}, {y}")
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Draw Lines", frame_copy)

        # Если 2 точки — рисуем первую линию (start_line)
        if len(points) == 2:
            cv2.line(frame_copy, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Draw Lines", frame_copy)
        # Если 4 точки — рисуем вторую (end_line)
        elif len(points) == 4:
            cv2.line(frame_copy, points[2], points[3], (255, 0, 0), 2)
            cv2.imshow("Draw Lines", frame_copy)
            print("\n✅ Линии нарисованы. Нажмите любую клавишу, чтобы сохранить.")


def main(video_path, output_yaml):
    global frame_copy

    # Загружаем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("❌ Не удалось прочитать кадр.")
        return

    frame_copy = frame.copy()
    cv2.imshow("Draw Lines", frame_copy)
    cv2.setMouseCallback("Draw Lines", click_event)

    print("🎯 Инструкция:")
    print("  1. Кликните 2 точки для первой линии (start_line — зелёная)")
    print("  2. Затем 2 точки для второй линии (end_line — синяя)")
    print("  3. После 4 кликов — нажмите любую клавишу для сохранения")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 4:
        print("⚠️ Нужно 4 точки (2 линии). Повторите попытку.")
        return

    data = {
        "start_line": [points[0], points[1]],
        "end_line": [points[2], points[3]]
    }

    # Сохраняем YAML
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"\n💾 Сохранено в: {output_yaml}")
    print(f"📄 Данные:\n{yaml.dump(data, sort_keys=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Интерактивное рисование линий для измерения скорости")
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    parser.add_argument("--output", default="line_zones.yaml", help="Путь для сохранения YAML (по умолчанию line_zones.yaml)")
    args = parser.parse_args()

    main(args.video, args.output)
