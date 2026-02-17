# speed_homography.py
# Измерение скорости через гомографию (перспектива → реальные метры)
# Работает с камерами под любым углом, учитывает перестроения

import cv2
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
import time
from collections import deque


class HomographySpeedEstimator:
    """
    Измеряет скорость автомобилей через гомографию.

    Преобразует координаты из пикселей в реальные метры,
    затем вычисляет скорость по пройденному пути.
    """

    def __init__(
        self,
        homography_config: str = None,
        H_matrix: np.ndarray = None,
        fps: float = 30.0,
        scale_px_per_m: float = 100.0,
        min_track_points: int = 5,
        smoothing_window: int = 10,
        max_speed_kmh: float = 200.0,
        min_speed_kmh: float = 5.0,
        speed_correction: float = 1.0,
    ):
        """
        Args:
            homography_config: путь к YAML с калибровкой
            H_matrix: матрица гомографии напрямую (если нет конфига)
            fps: частота кадров
            scale_px_per_m: масштаб (пикселей на метр в bird's eye)
            min_track_points: минимум точек для расчёта скорости
            smoothing_window: окно сглаживания (последние N точек)
            max_speed_kmh: максимальная разумная скорость (фильтр выбросов)
            min_speed_kmh: минимальная скорость (ниже — стоит)
            speed_correction: коэффициент коррекции скорости (для тонкой настройки)
        """
        self.fps = fps
        self.scale = scale_px_per_m
        self.min_track_points = min_track_points
        self.smoothing_window = smoothing_window
        self.max_speed_kmh = max_speed_kmh
        self.min_speed_kmh = min_speed_kmh
        self.speed_correction = speed_correction
        print(f"   Speed correction: {speed_correction}")

        # Загружаем матрицу гомографии
        if homography_config:
            self._load_config(homography_config)
        elif H_matrix is not None:
            self.H = np.array(H_matrix, dtype=np.float64)
        else:
            raise ValueError("Нужен homography_config или H_matrix")

        # Треки: {obj_id: deque([(frame_idx, x_m, y_m), ...])}
        self.tracks: Dict[int, deque] = {}

        # Последние скорости: {obj_id: speed_kmh}
        self.speeds: Dict[int, float] = {}

        # Для визуализации: {obj_id: deque([(x_m, y_m), ...], maxlen=100)}
        self.paths: Dict[int, deque] = {}

        # Кэш bird eye view (обновляется раз в N кадров)
        self._bev_cache: Optional[np.ndarray] = None
        self._bev_cache_frame: int = 0
        self._bev_update_interval: int = 5  # обновлять каждые 5 кадров

    def _load_config(self, path: str):
        """Загружает конфиг калибровки"""
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        hom = cfg.get("homography", cfg)

        if "matrix" in hom:
            self.H = np.array(hom["matrix"], dtype=np.float64)
        else:
            # Загружаем из .npy файла
            matrix_file = path.replace(".yaml", "_matrix.npy")
            self.H = np.load(matrix_file)

        self.scale = hom.get("scale_px_per_m", self.scale)
        print(f"📐 Гомография загружена: {path}")
        print(f"   Масштаб: {self.scale} px/m")

    def transform_point(self, px: float, py: float) -> Tuple[float, float]:
        """
        Преобразует точку из пикселей камеры в реальные метры.

        Returns:
            (x_meters, y_meters)
        """
        pt = np.array([px, py, 1.0], dtype=np.float64)
        transformed = self.H @ pt

        if abs(transformed[2]) < 1e-10:
            return 0.0, 0.0

        x_px = transformed[0] / transformed[2]
        y_px = transformed[1] / transformed[2]

        # Переводим из "пикселей bird's eye" в метры
        x_m = x_px / self.scale
        y_m = y_px / self.scale

        return x_m, y_m

    def process(self, frame_idx: int, obj_id: int, px: float, py: float) -> Optional[float]:
        """
        Обновляет трек объекта и возвращает скорость.

        Args:
            frame_idx: номер кадра
            obj_id: ID объекта (из трекера)
            px, py: координаты в пикселях (центр bbox или низ)

        Returns:
            Скорость в км/ч или None если недостаточно данных
        """
        # Преобразуем в метры
        x_m, y_m = self.transform_point(px, py)

        # Инициализируем трек если нужно
        if obj_id not in self.tracks:
            self.tracks[obj_id] = deque(maxlen=self.smoothing_window * 2)
            self.paths[obj_id] = deque(maxlen=100)

        # Добавляем точку
        self.tracks[obj_id].append((frame_idx, x_m, y_m))
        self.paths[obj_id].append((x_m, y_m))

        # Проверяем достаточно ли точек
        if len(self.tracks[obj_id]) < self.min_track_points:
            return None

        # Вычисляем скорость
        speed = self._calculate_speed(obj_id)

        if speed is not None:
            self.speeds[obj_id] = speed

        return speed

    def _calculate_speed(self, obj_id: int) -> Optional[float]:
        """
        Вычисляет скорость по последним точкам трека.

        Использует суммарное пройденное расстояние, а не только
        начальную и конечную точку — это точнее при маневрах.
        """
        track = list(self.tracks[obj_id])

        # Берём последние N точек
        n = min(len(track), self.smoothing_window)
        points = track[-n:]

        if len(points) < 2:
            return None

        # Суммируем пройденное расстояние
        total_distance = 0.0
        for i in range(1, len(points)):
            dx = points[i][1] - points[i - 1][1]
            dy = points[i][2] - points[i - 1][2]
            segment = np.sqrt(dx * dx + dy * dy)
            total_distance += segment

        # Время между первой и последней точкой
        dt_frames = points[-1][0] - points[0][0]
        if dt_frames <= 0:
            return None

        dt_seconds = dt_frames / self.fps

        # Скорость в м/с → км/ч
        speed_ms = total_distance / dt_seconds
        speed_kmh = speed_ms * 3.6 * self.speed_correction

        # Фильтруем выбросы
        if speed_kmh > self.max_speed_kmh:
            return None
        if speed_kmh < self.min_speed_kmh:
            return 0.0

        return speed_kmh

    def get_speed(self, obj_id: int) -> Optional[float]:
        """Возвращает последнюю известную скорость объекта"""
        return self.speeds.get(obj_id)

    def get_path_meters(self, obj_id: int) -> List[Tuple[float, float]]:
        """Возвращает путь объекта в метрах (для визуализации)"""
        return self.paths.get(obj_id, [])

    def draw_info(self, frame: np.ndarray, x1: int, y1: int, obj_id: int):
        """Рисует скорость над bbox"""
        speed = self.speeds.get(obj_id)
        if speed is not None:
            color = (0, 255, 0) if speed < 60 else (0, 165, 255) if speed < 90 else (0, 0, 255)
            label = f"{speed:.0f} km/h"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def draw_bird_eye_view(
        self,
        size: Tuple[int, int] = (400, 600),
        margin: int = 50,
        show_grid: bool = True,
        grid_step_m: float = 5.0,
        frame_idx: int = 0
    ) -> np.ndarray:
        """
        Создаёт миникарту Bird's Eye View с треками.
        Использует кэш для ускорения (обновляется раз в N кадров).

        Returns:
            Изображение numpy array (BGR)
        """
        # Проверяем кэш
        if (self._bev_cache is not None and
            frame_idx - self._bev_cache_frame < self._bev_update_interval):
            return self._bev_cache

        w, h = size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Определяем границы (автоматически по всем трекам)
        all_points = []
        for path in self.paths.values():
            all_points.extend(list(path)[-50:])

        if not all_points:
            return canvas

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        x_min, x_max = min(xs) - 2, max(xs) + 2
        y_min, y_max = min(ys) - 2, max(ys) + 2

        # Масштаб для отображения
        scale_x = (w - 2 * margin) / max(x_max - x_min, 0.1)
        scale_y = (h - 2 * margin) / max(y_max - y_min, 0.1)
        scale = min(scale_x, scale_y)

        def to_canvas(xm, ym):
            cx = int(margin + (xm - x_min) * scale)
            cy = int(margin + (ym - y_min) * scale)
            return cx, cy

        # Сетка
        if show_grid:
            for x in np.arange(np.floor(x_min), np.ceil(x_max), grid_step_m):
                cx, _ = to_canvas(x, 0)
                cv2.line(canvas, (cx, 0), (cx, h), (40, 40, 40), 1)
            for y in np.arange(np.floor(y_min), np.ceil(y_max), grid_step_m):
                _, cy = to_canvas(0, y)
                cv2.line(canvas, (0, cy), (w, cy), (40, 40, 40), 1)

        # Треки
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for idx, (obj_id, path) in enumerate(self.paths.items()):
            if len(path) < 2:
                continue

            color = colors[idx % len(colors)]
            recent = list(path)[-30:]

            # Линия пути
            pts = [to_canvas(p[0], p[1]) for p in recent]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 2)

            # Текущая позиция
            cv2.circle(canvas, pts[-1], 6, color, -1)

            # Скорость
            speed = self.speeds.get(obj_id)
            if speed is not None:
                cv2.putText(canvas, f"{speed:.0f}", (pts[-1][0] + 10, pts[-1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Заголовок
        cv2.putText(canvas, "Bird's Eye View", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Сохраняем в кэш
        self._bev_cache = canvas
        self._bev_cache_frame = frame_idx

        return canvas

    def cleanup_old_tracks(self, current_frame: int, max_age_frames: int = 150):
        """Удаляет старые треки (объекты ушли из кадра)"""
        to_remove = []
        for obj_id, track in self.tracks.items():
            if track and current_frame - track[-1][0] > max_age_frames:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracks[obj_id]
            self.speeds.pop(obj_id, None)
            self.paths.pop(obj_id, None)
