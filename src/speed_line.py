# speed_line_fps.py
# Модуль трекинга машин и измерения скорости по линиям.
# Используется из main_anpr_speed.py.

import cv2
import numpy as np


class SpeedLineZone:
    """Пара линий (start_line, end_line) и расстояние между ними в метрах."""

    def __init__(self, start_line, end_line, distance_m,
                 direction="down", name="Zone", color=(0, 255, 0)):
        self.start_line = start_line
        self.end_line = end_line
        self.distance_m = distance_m
        self.name = name
        self.color = color
        self.direction = direction  # "up" или "down"

    @property
    def y1(self):
        return (self.start_line[0][1] + self.start_line[1][1]) / 2

    @property
    def y2(self):
        return (self.end_line[0][1] + self.end_line[1][1]) / 2


# =====================================================
# Класс измерения скорости (через зоны)
# =====================================================
class SpeedEstimator:
    """
    Измеряет скорость между двумя линиями.
    Скорость выводится только после выхода из зоны.
    OCR запускается через self.post_time_sec секунд после выхода.
    """

    def __init__(self, fps):
        self.fps = fps
        self.zones = []
        self.entry_frames = {}     # {id: {zone: frame_idx}}
        self.object_speeds = {}    # {id: (v, color)}
        self.ocr_trigger = {}      # {id: frame_idx_when_trigger_OCR}
        self.prev_positions = {}   # {id: prev_cy}
        self.smooth_speeds = {}    # для сглаживания
        self.post_time_sec = 0.3     # задержка OCR после выхода из зоны

    # -------------------------------------------------
    def add_line_zone(self, start_line, end_line, distance_m,
                      direction="down", name="Zone", color=(0, 255, 0)):
        """Добавляем зону в список."""
        self.zones.append(SpeedLineZone(start_line, end_line,
                                        distance_m, direction, name, color))

    # -------------------------------------------------
    # Основной расчёт скорости
    # -------------------------------------------------
    def process(self, frame_idx, obj_id, cx, cy):
        """Проверка пересечения линий и расчёт скорости."""
        prev_cy = self.prev_positions.get(obj_id, cy)
        self.prev_positions[obj_id] = cy

        for zone in self.zones:
            # --- Проверяем, пересек ли объект линии ---
            if zone.direction == "down":
                cond_enter = (prev_cy < zone.y1 <= cy)
                cond_exit = (prev_cy < zone.y2 <= cy)
            else:  # движение вверх
                cond_enter = (prev_cy > zone.y1 >= cy)
                cond_exit = (prev_cy > zone.y2 >= cy)

            # --- вход в зону ---
            if cond_enter:
                self.entry_frames.setdefault(obj_id, {})[zone.name] = frame_idx

            # --- выход из зоны ---
            if cond_exit and obj_id in self.entry_frames and zone.name in self.entry_frames[obj_id]:
                f1 = self.entry_frames[obj_id][zone.name]
                dt = (frame_idx - f1) / self.fps
                if dt > 0:
                    v_kmh = (zone.distance_m / dt) * 3.6
                    v_kmh = self._smooth_speed(obj_id, v_kmh)
                    # 🚗 фиксируем скорость только после второй линии
                    self.object_speeds[obj_id] = (v_kmh, zone.color)
                    # 🕒 планируем OCR через 2 секунды после выхода
                    self.ocr_trigger[obj_id] = frame_idx
                    print(f"✅ ID {obj_id}: {v_kmh:.1f} km/h (zone={zone.name}, delay={self.post_time_sec}s)")
                # очищаем вход, чтобы не считать повторно
                del self.entry_frames[obj_id][zone.name]

    # -------------------------------------------------
    def _smooth_speed(self, obj_id, new_speed, alpha=0.3):
        """Сглаживание скорости."""
        prev = self.smooth_speeds.get(obj_id, new_speed)
        smoothed = prev * (1 - alpha) + new_speed * alpha
        self.smooth_speeds[obj_id] = smoothed
        return smoothed

    # -------------------------------------------------
    # Визуализация
    # -------------------------------------------------
    def draw_zones(self, frame):
        """Рисуем линии зон и подписи."""
        for zone in self.zones:
            cv2.line(frame, tuple(zone.start_line[0]), tuple(zone.start_line[1]), zone.color, 2)
            cv2.line(frame, tuple(zone.end_line[0]), tuple(zone.end_line[1]), zone.color, 2)
            cv2.putText(frame, zone.name,
                        (zone.start_line[0][0] + 5, int(zone.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone.color, 2)

    def draw_info(self, frame, x1, y1, obj_id):
        """Подпись ID и (если уже рассчитана) скорости."""
        if obj_id in self.object_speeds:
            v_kmh, color = self.object_speeds[obj_id]
            label = f"ID {obj_id} | {v_kmh:.1f} km/h"
        else:
            label = f"ID {obj_id}"
            color = (180, 180, 180)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

      # -------------------------------------------------
    # Красивый оверлей номера и скорости
    # -------------------------------------------------
    def draw_plate_info(self, frame, x1, y1, obj_id, plate_text=None):
        """
        Отображает "плашку" с номером и скоростью над боксом.
        plate_text — строка номера (или None).
        """
        v_kmh, color = self.object_speeds.get(obj_id, (None, (200, 200, 200)))
        if v_kmh is None and plate_text is None:
            return  # нечего рисовать

        # позиция блока
        start_x = int(x1)
        start_y = int(y1) - 55
        w, h = 180, 50
        end_x, end_y = start_x + w, start_y + h

        # фон с прозрачностью
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # рамка
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        # текст: номер
        text_plate = plate_text if plate_text else "..."
        cv2.putText(frame, text_plate,
                    (start_x + 8, start_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # текст: скорость
        if v_kmh is not None:
            cv2.putText(frame, f"{v_kmh:.1f} km/h",
                        (start_x + 8, start_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)