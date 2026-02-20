# speed_homography.py
# Измерение скорости через гомографию (перспектива → реальные метры)
#
# Алгоритм:
#   1. Каждый кадр: пиксель (cx, cy) → гомография → реальные метры (xm, ym)
#   2. Трек = deque точек (frame, xm, ym) для каждого obj_id
#   3. Скорость = net displacement за скользящее окно / время
#   4. Фильтры применяются к RAW скорости (до коррекции):
#      - min_raw_speed: отсекает стоящие объекты (jitter трекера)
#      - max_raw_speed: отсекает скачки трекера (ID swap, телепортация)
#      - jump detection: сброс трека при невозможном перемещении
#   5. speed_correction применяется ТОЛЬКО к итоговому результату
#   6. Медиана всех измерений = стабильная скорость для отчёта

import cv2
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from collections import deque


class HomographySpeedEstimator:
    """
    Измеряет скорость автомобилей через гомографию.

    Пиксели камеры → bird's eye view (метры) → скорость по пройденному пути.
    speed_correction — поправочный коэффициент калибровки (умножает итог).
    """

    def __init__(
        self,
        homography_config: str = None,
        H_matrix: np.ndarray = None,
        fps: float = 25.0,
        scale_px_per_m: float = 100.0,
        min_track_points: int = 5,
        smoothing_window: int = 10,
        max_speed_kmh: float = 200.0,
        min_speed_kmh: float = 5.0,
        speed_correction: float = 1.0,
    ):
        self.fps = fps
        self.scale = scale_px_per_m
        self.min_track_points = min_track_points
        self.smoothing_window = smoothing_window
        self.speed_correction = speed_correction

        # Фильтры — всегда применяются к RAW (до коррекции)
        self.min_raw_speed = min_speed_kmh
        self.max_raw_speed = max_speed_kmh

        # Порог скачка: если между двумя кадрами raw > jump_threshold → ID swap
        self.jump_threshold_kmh = 500.0

        print(f"   Speed correction: {speed_correction}")

        # Загружаем матрицу гомографии
        if homography_config:
            self._load_config(homography_config)
        elif H_matrix is not None:
            self.H = np.array(H_matrix, dtype=np.float64)
        else:
            raise ValueError("Нужен homography_config или H_matrix")

        # --- Per-track state ---
        # Точки трека: {obj_id: deque([(frame, xm, ym), ...])}
        self.tracks: Dict[int, deque] = {}

        # Текущая скорость (corrected) для OSD
        self.speeds: Dict[int, float] = {}

        # История RAW измерений для медианы: {obj_id: list[float]}
        self.raw_history: Dict[int, List[float]] = {}

        # Путь в метрах для BEV: {obj_id: deque([(xm, ym)], maxlen=100)}
        self.paths: Dict[int, deque] = {}

        # BEV cache
        self._bev_cache: Optional[np.ndarray] = None
        self._bev_cache_frame: int = 0
        self._bev_update_interval: int = 5

    # ------------------------------------------------------------------
    #  Config
    # ------------------------------------------------------------------

    def _load_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        hom = cfg.get("homography", cfg)

        if "matrix" in hom:
            self.H = np.array(hom["matrix"], dtype=np.float64)
        else:
            matrix_file = path.replace(".yaml", "_matrix.npy")
            self.H = np.load(matrix_file)

        self.scale = hom.get("scale_px_per_m", self.scale)
        print(f"📐 Гомография загружена: {path}")
        print(f"   Масштаб: {self.scale} px/m")

    # ------------------------------------------------------------------
    #  Transform
    # ------------------------------------------------------------------

    def transform_point(self, px: float, py: float) -> Tuple[float, float]:
        """Пиксели камеры → метры (bird's eye)."""
        pt = np.array([px, py, 1.0], dtype=np.float64)
        t = self.H @ pt

        if abs(t[2]) < 1e-10:
            return 0.0, 0.0

        return (t[0] / t[2]) / self.scale, (t[1] / t[2]) / self.scale

    # ------------------------------------------------------------------
    #  Core: process() — вызывается каждый кадр для каждого объекта
    # ------------------------------------------------------------------

    def process(
        self, frame_idx: int, obj_id: int, px: float, py: float
    ) -> Optional[float]:
        """
        Обновляет трек и возвращает скорость (corrected, км/ч) или None.

        Args:
            frame_idx: номер кадра
            obj_id:    ID объекта (из ByteTrack)
            px, py:    координаты в пикселях (bottom-center bbox)
        """
        xm, ym = self.transform_point(px, py)

        # --- Инициализация нового трека ---
        if obj_id not in self.tracks:
            self.tracks[obj_id] = deque(maxlen=self.smoothing_window * 2)
            self.paths[obj_id] = deque(maxlen=100)
            self.raw_history[obj_id] = []

        # --- Jump detection: сброс при невозможном скачке ---
        track = self.tracks[obj_id]
        if len(track) > 0:
            prev_frame, prev_xm, prev_ym = track[-1]
            dt_frames = frame_idx - prev_frame
            if dt_frames > 0:
                dx = xm - prev_xm
                dy = ym - prev_ym
                dt_sec = dt_frames / self.fps
                instant_raw = np.sqrt(dx * dx + dy * dy) / dt_sec * 3.6
                if instant_raw > self.jump_threshold_kmh:
                    # Скачок — вероятно ID swap, сбрасываем трек
                    track.clear()
                    self.paths[obj_id].clear()
                    self.raw_history[obj_id].clear()

        # --- Добавляем точку ---
        track.append((frame_idx, xm, ym))
        self.paths[obj_id].append((xm, ym))

        # --- Недостаточно точек ---
        if len(track) < self.min_track_points:
            return None

        # --- Расчёт скорости ---
        raw_kmh = self._calc_raw_speed(obj_id)
        if raw_kmh is None:
            return None

        # Сохраняем raw в историю (для медианы)
        self.raw_history[obj_id].append(raw_kmh)

        # Применяем коррекцию
        corrected = raw_kmh * self.speed_correction
        self.speeds[obj_id] = corrected

        return corrected

    # ------------------------------------------------------------------
    #  Расчёт RAW скорости (до коррекции)
    # ------------------------------------------------------------------

    def _calc_raw_speed(self, obj_id: int) -> Optional[float]:
        """
        Net displacement за скользящее окно → raw км/ч.

        Net displacement (first→last) а не сумма отрезков —
        устойчиво к jitter трекера.
        Возвращает None если не проходит фильтры.
        """
        track = self.tracks[obj_id]
        n = min(len(track), self.smoothing_window)
        pts = list(track)[-n:]

        if len(pts) < 2:
            return None

        # Net displacement
        dx = pts[-1][1] - pts[0][1]
        dy = pts[-1][2] - pts[0][2]
        dist_m = np.sqrt(dx * dx + dy * dy)

        dt_frames = pts[-1][0] - pts[0][0]
        if dt_frames <= 0:
            return None

        dt_sec = dt_frames / self.fps
        raw_kmh = (dist_m / dt_sec) * 3.6

        # Фильтры на RAW (до коррекции)
        if raw_kmh > self.max_raw_speed:
            return None
        if raw_kmh < self.min_raw_speed:
            return None

        return raw_kmh

    # ------------------------------------------------------------------
    #  Getters
    # ------------------------------------------------------------------

    def get_speed(self, obj_id: int) -> Optional[float]:
        """Последняя известная скорость (corrected) для OSD / speed_tracker."""
        return self.speeds.get(obj_id)

    def get_median_speed(self, obj_id: int) -> Optional[float]:
        """Медиана всех RAW измерений × correction. Стабильнее чем max/last."""
        hist = self.raw_history.get(obj_id)
        if not hist:
            return None
        return float(np.median(hist)) * self.speed_correction

    def get_all_median_speeds(self) -> Dict[int, float]:
        """Медианные скорости для всех треков."""
        result = {}
        for obj_id in self.raw_history:
            v = self.get_median_speed(obj_id)
            if v is not None:
                result[obj_id] = v
        return result

    def get_path_meters(self, obj_id: int) -> List[Tuple[float, float]]:
        """Путь объекта в метрах (для визуализации)."""
        return list(self.paths.get(obj_id, []))

    # ------------------------------------------------------------------
    #  Визуализация
    # ------------------------------------------------------------------

    def draw_info(self, frame: np.ndarray, x1: int, y1: int, obj_id: int):
        """Рисует скорость над bbox."""
        speed = self.speeds.get(obj_id)
        if speed is not None:
            if speed < 60:
                color = (0, 255, 0)
            elif speed < 90:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.putText(
                frame, f"{speed:.0f} km/h", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )

    def draw_bird_eye_view(
        self,
        size: Tuple[int, int] = (400, 600),
        margin: int = 50,
        show_grid: bool = True,
        grid_step_m: float = 5.0,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Миникарта Bird's Eye View с треками (кэшируется)."""
        if (
            self._bev_cache is not None
            and frame_idx - self._bev_cache_frame < self._bev_update_interval
        ):
            return self._bev_cache

        w, h = size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        all_points = []
        for path in self.paths.values():
            all_points.extend(list(path)[-50:])

        if not all_points:
            return canvas

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        x_min, x_max = min(xs) - 2, max(xs) + 2
        y_min, y_max = min(ys) - 2, max(ys) + 2

        scale_x = (w - 2 * margin) / max(x_max - x_min, 0.1)
        scale_y = (h - 2 * margin) / max(y_max - y_min, 0.1)
        sc = min(scale_x, scale_y)

        def to_canvas(xm, ym):
            return (
                int(margin + (xm - x_min) * sc),
                int(margin + (ym - y_min) * sc),
            )

        if show_grid:
            for x in np.arange(np.floor(x_min), np.ceil(x_max), grid_step_m):
                cx, _ = to_canvas(x, 0)
                cv2.line(canvas, (cx, 0), (cx, h), (40, 40, 40), 1)
            for y in np.arange(np.floor(y_min), np.ceil(y_max), grid_step_m):
                _, cy = to_canvas(0, y)
                cv2.line(canvas, (0, cy), (w, cy), (40, 40, 40), 1)

        colors = [
            (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0),
        ]
        for idx, (oid, path) in enumerate(self.paths.items()):
            if len(path) < 2:
                continue
            color = colors[idx % len(colors)]
            recent = list(path)[-30:]
            pts = [to_canvas(p[0], p[1]) for p in recent]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 2)
            cv2.circle(canvas, pts[-1], 6, color, -1)
            spd = self.speeds.get(oid)
            if spd is not None:
                cv2.putText(
                    canvas, f"{spd:.0f}",
                    (pts[-1][0] + 10, pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        cv2.putText(
            canvas, "Bird's Eye View", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        self._bev_cache = canvas
        self._bev_cache_frame = frame_idx
        return canvas

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_tracks(self, current_frame: int, max_age_frames: int = 150):
        """Удаляет треки объектов, ушедших из кадра."""
        to_remove = []
        for obj_id, track in self.tracks.items():
            if track and current_frame - track[-1][0] > max_age_frames:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracks[obj_id]
            self.speeds.pop(obj_id, None)
            self.paths.pop(obj_id, None)
            self.raw_history.pop(obj_id, None)
