# plate_recognizer.py
# OCR номеров через NomeroffNet
# Сохранение: passed/ (прошли фильтры) и failed/ (лучшие из не прошедших)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import os
import json
import re
import time
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip


@dataclass
class PlateEvent:
    """Полные данные о распознавании"""
    event_id: str = ""
    timestamp: str = ""
    camera_id: str = ""
    track_id: int = 0
    frame_idx: int = 0

    # Три раздельных score (0-1)
    car_score: float = 0.0       # Уверенность что это машина (YOLO)
    plate_score: float = 0.0     # Качество crop номера (blur, brightness, size)
    ocr_score: float = 0.0       # Качество OCR текста (длина, формат)
    total_score: float = 0.0     # Среднее трёх scores

    # Legacy (для совместимости)
    detection_conf: float = 0.0  # = car_score (YOLO raw)
    plate_conf: float = 0.0      # уверенность детектора номера NomeroffNet
    ocr_conf: float = 0.0        # = total_score (для совместимости)
    processing_time_ms: float = 0.0

    # Результат
    plate_text: str = ""
    region: str = ""

    # Raw данные
    brightness: float = 0.0      # средняя яркость
    blur: float = 0.0            # Laplacian variance (резкость)
    plate_width_px: int = 0
    plate_height_px: int = 0
    car_width_px: int = 0
    car_height_px: int = 0

    # Пути к изображениям
    crop_path: str = ""

    # Причина отклонения (для failed)
    reject_reason: str = ""

    # Кроп (не сохраняется в JSON)
    crop: np.ndarray = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Конвертация в словарь для JSON (без crop)"""
        d = asdict(self)
        del d['crop']
        # Конвертируем numpy типы в Python типы
        for key, value in d.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                d[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                d[key] = float(value)
        return d


class PlateRecognizer:
    """
    Распознаватель номеров с организованным выводом.

    Структура папок:
    output_dir/
    ├── passed/          # Прошли все фильтры
    │   ├── images/
    │   └── results.json
    └── failed/          # Лучшие из не прошедших (по формату)
        ├── images/
        └── results.json
    """

    def __init__(
        self,
        output_dir: str,
        camera_id: str = "camera_01",
        min_conf: float = 0.5,
        min_plate_chars: int = 8,
        min_car_height: int = 150,
        min_car_width: int = 100,
        min_plate_width: int = 60,
        min_plate_height: int = 15,
        cooldown_frames: int = 3,
        plate_format_regex: str = "",
        file_logger=None,  # FileLogger для записи в файлы
    ):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.min_conf = min_conf
        self.min_plate_chars = min_plate_chars
        self.min_car_height = min_car_height
        self.min_car_width = min_car_width
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height
        self.cooldown_frames = cooldown_frames
        self.plate_format_regex = plate_format_regex
        self.file_logger = file_logger

        # Создаём структуру папок
        self.passed_dir = os.path.join(output_dir, "passed")
        self.passed_images = os.path.join(self.passed_dir, "images")
        self.failed_dir = os.path.join(output_dir, "failed")
        self.failed_images = os.path.join(self.failed_dir, "images")

        os.makedirs(self.passed_images, exist_ok=True)
        os.makedirs(self.failed_images, exist_ok=True)

        # NomeroffNet с оптимизацией
        print("Loading NomeroffNet...")
        self.pipeline = pipeline(
            "number_plate_detection_and_reading_runtime",
            off_number_plate_classification=True,  # отключаем классификацию (~10мс экономия)
            default_label="kz",                    # Казахстан
            default_lines_count=1,
            path_to_model=r"C:\Users\user\Desktop\speed_limit\nomeroff-net\data\models\Detector\yolov11x\yolov11x-keypoints-2024-10-11.engine",
        )
        print("NomeroffNet ready (yolov11x-keypoints TensorRT, classification OFF, default=kz)")

        # Оптимизация: максимальный размер входа для локализации
        self.max_ocr_width = 320  # уменьшаем кропы (было 480) — быстрее YOLO

        # Лучшие результаты
        self.passed_results: Dict[int, PlateEvent] = {}  # прошли все фильтры
        self.failed_results: Dict[int, PlateEvent] = {}  # не прошли формат, но лучшие

        # Двухуровневая оптимизация
        self.best_quality: Dict[int, float] = {}  # лучший quality score по track_id
        self.last_ocr_frame: Dict[int, int] = {}  # cooldown

        # Пороги качества для дешёвого отбора
        self.min_blur_score = 50.0      # минимальная резкость
        self.min_brightness = 40.0       # минимальная яркость
        self.max_brightness = 220.0      # максимальная яркость (пересвет)
        self.quality_improvement = 1.15  # на сколько должен улучшиться quality (15%)

        # Статистика
        self.stats = {
            "total_frames": 0,
            "skipped_car_size": 0,
            "skipped_quality": 0,
            "skipped_not_better": 0,
            "skipped_cooldown": 0,
            "ocr_called": 0,
            "ocr_no_plate": 0,
            "skipped_plate_size": 0,
            "skipped_chars": 0,
            "skipped_format": 0,
            "passed": 0,
        }

        # Последние метрики качества (для логирования)
        self.last_blur: float = 0.0
        self.last_brightness: float = 0.0


    def _calculate_brightness(self, img: np.ndarray) -> float:
        """Средняя яркость изображения"""
        if img is None or img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(np.mean(gray))

    def _calculate_blur(self, img: np.ndarray) -> float:
        """Оценка резкости (Laplacian variance). Больше = резче"""
        if img is None or img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _calculate_quality(self, img: np.ndarray, car_width: int, car_height: int) -> Tuple[float, float, float]:
        """
        Быстрый расчёт quality score (микросекунды).
        Returns: (quality_score, blur, brightness)
        """
        if img is None or img.size == 0:
            return 0.0, 0.0, 0.0

        # Быстрый blur на уменьшенном изображении
        h, w = img.shape[:2]
        scale = 64 / max(h, w)  # уменьшаем до ~64px (быстрее)
        if scale < 1.0:
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        else:
            small = img

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        # Quality score: комбинация факторов
        # - blur: больше = лучше (резче)
        # - size: больше = лучше (машина ближе)
        # - brightness: оптимум ~120, штраф за пересвет/темноту
        size_score = (car_width * car_height) / 10000  # нормализация
        brightness_penalty = 1.0 - abs(brightness - 120) / 120  # оптимум 120
        brightness_penalty = max(0.1, brightness_penalty)

        quality = blur * size_score * brightness_penalty

        return quality, blur, brightness

    def _is_quality_acceptable(self, blur: float, brightness: float) -> Tuple[bool, str]:
        """Быстрая проверка минимального качества"""
        if blur < self.min_blur_score:
            return False, f"blur:{blur:.0f}<{self.min_blur_score}"
        if brightness < self.min_brightness:
            return False, f"dark:{brightness:.0f}"
        if brightness > self.max_brightness:
            return False, f"bright:{brightness:.0f}"
        return True, ""

    def _calculate_scores(
        self,
        blur: float,
        brightness: float,
        plate_width: int,
        plate_height: int,
        detection_conf: float,
        text_len: int,
        format_valid: bool = True,
    ) -> Dict[str, float]:
        """
        Три отдельных score (0.0 - 1.0):

        1. car_score: уверенность что это машина (YOLO detection_conf)
        2. plate_score: качество crop номера (blur, brightness, size)
        3. ocr_score: качество OCR текста (длина, формат)

        Returns: {car_score, plate_score, ocr_score, total}
        """
        # === 1. CAR SCORE: уверенность детекции ===
        car_score = max(0.5, min(1.0, detection_conf))

        # === 2. PLATE SCORE: качество изображения номера ===
        # Blur (резкость): 50 = плохо (0.3), 300+ = отлично (1.0)
        blur_norm = min(1.0, max(0.3, (blur - 50) / 250 + 0.3))

        # Brightness (яркость): оптимум 80-160
        if 80 <= brightness <= 160:
            bright_norm = 1.0
        elif 60 <= brightness < 80 or 160 < brightness <= 200:
            bright_norm = 0.7
        else:
            bright_norm = 0.4

        # Size (размер номера): 60px = мин (0.5), 150px+ = макс (1.0)
        size_norm = min(1.0, max(0.5, (plate_width - 60) / 180 + 0.5))

        plate_score = (blur_norm * 0.4 + bright_norm * 0.2 + size_norm * 0.4)

        # === 3. OCR SCORE: качество распознанного текста ===
        # Длина текста: 8 символов = идеально
        if text_len >= 8:
            len_norm = 1.0
        elif text_len >= 6:
            len_norm = 0.7
        elif text_len >= 4:
            len_norm = 0.4
        else:
            len_norm = 0.2

        # Формат: соответствует regex = +0.3
        format_norm = 1.0 if format_valid else 0.6

        ocr_score = (len_norm * 0.6 + format_norm * 0.4)

        # === TOTAL: среднее трёх scores ===
        total = (car_score + plate_score + ocr_score) / 3

        return {
            "car_score": round(car_score, 3),
            "plate_score": round(plate_score, 3),
            "ocr_score": round(ocr_score, 3),
            "total": round(total, 3),
        }

    def _detect_plate(self, car_crop: np.ndarray) -> Optional[Tuple[str, float, float, int, int, str]]:
        """
        Распознаёт номер через NomeroffNet.
        Returns: (text, plate_conf, ocr_conf, plate_width, plate_height, region) или None
        """
        if car_crop is None or car_crop.size == 0:
            return None

        try:
            h, w = car_crop.shape[:2]
            scale = 1.0
            if w > self.max_ocr_width:
                scale = self.max_ocr_width / w
                car_crop = cv2.resize(car_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
            res = self.pipeline([rgb])

            if isinstance(res, tuple) and len(res) > 1:
                data = res[1][0] if res[1] else None
            elif isinstance(res, list) and len(res) > 0:
                data = res[0]
            else:
                return None

            if not isinstance(data, (list, tuple)) or len(data) < 9:
                return None

            bboxs = data[1]
            texts = data[8]
            confidences = data[7]
            regions = data[5] if len(data) > 5 else []

            if not bboxs or len(bboxs) == 0:
                return None

            # Bbox номера (пересчитываем обратно если был resize)
            bbox = bboxs[0]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            plate_width = int((x2 - x1) / scale)  # оригинальный размер
            plate_height = int((y2 - y1) / scale)

            # Проверка размера ДО OCR
            if plate_width < self.min_plate_width or plate_height < self.min_plate_height:
                self.stats["skipped_plate_size"] += 1
                return None

            # Текст
            text = texts[0] if texts and len(texts) > 0 else ""
            text = text.replace(" ", "").upper()

            # Регион
            region = regions[0] if regions and len(regions) > 0 else "unknown"

            # Confidence детектора
            plate_conf = 0.95

            # ocr_conf будет рассчитан в process() на основе всех факторов
            ocr_conf = 0.0  # placeholder

            return (text, plate_conf, ocr_conf, plate_width, plate_height, region)

        except Exception as e:
            print(f"OCR error: {e}")
            return None

    def process(
        self,
        track_id: int,
        car_crop: np.ndarray,
        car_height: int = 0,
        car_width: int = 0,
        bbox: Tuple[int, int, int, int] = None,
        frame_idx: int = 0,
        detection_conf: float = 0.0,
    ) -> Optional[PlateEvent]:
        """
        Обрабатывает кроп машины.
        Двухуровневая оптимизация:
        1. Дешёвый отбор (микросекунды) - blur, brightness, size
        2. Дорогой pipeline только для "претендентов"
        """

        self.stats["total_frames"] += 1

        if car_crop is None or car_crop.size == 0:
            return self._get_best_result(track_id)

        # === ФИЛЬТР 1: Размер машины (дешёвый) ===
        if car_height > 0 and car_height < self.min_car_height:
            self.stats["skipped_car_size"] += 1
            return self._get_best_result(track_id)
        if car_width > 0 and car_width < self.min_car_width:
            self.stats["skipped_car_size"] += 1
            return self._get_best_result(track_id)

        # === ФИЛЬТР 2: Качество кадра (дешёвый, ~0.1мс) ===
        quality, blur, brightness = self._calculate_quality(car_crop, car_width, car_height)

        # Сохраняем для логирования
        self.last_blur = blur
        self.last_brightness = brightness

        acceptable, reject_reason = self._is_quality_acceptable(blur, brightness)
        if not acceptable:
            self.stats["skipped_quality"] += 1
            return self._get_best_result(track_id)

        # === ФИЛЬТР 3: Кадр должен быть лучше предыдущего ===
        best_q = self.best_quality.get(track_id, 0.0)
        if quality < best_q * self.quality_improvement:
            # Кадр не достаточно лучше — пропускаем дорогой pipeline
            self.stats["skipped_not_better"] += 1
            return self._get_best_result(track_id)

        # === ФИЛЬТР 4: Cooldown (минимальный интервал) ===
        last_frame = self.last_ocr_frame.get(track_id, -999)
        if frame_idx - last_frame < self.cooldown_frames:
            self.stats["skipped_cooldown"] += 1
            return self._get_best_result(track_id)

        # Кадр — претендент! Запускаем дорогой pipeline
        self.last_ocr_frame[track_id] = frame_idx
        self.best_quality[track_id] = quality  # обновляем лучший score
        self.stats["ocr_called"] += 1

        start_time = time.time()
        plate_info = self._detect_plate(car_crop)
        processing_time = (time.time() - start_time) * 1000

        if plate_info is None:
            self.stats["ocr_no_plate"] += 1
            return self._get_best_result(track_id)

        text, plate_conf, _, plate_width, plate_height, region = plate_info

        # Проверяем формат для ocr_score
        format_valid = True
        if self.plate_format_regex:
            import re
            format_valid = bool(re.match(self.plate_format_regex, text))

        # Вычисляем три отдельных score
        scores = self._calculate_scores(
            blur=blur,
            brightness=brightness,
            plate_width=plate_width,
            plate_height=plate_height,
            detection_conf=detection_conf,
            text_len=len(text),
            format_valid=format_valid,
        )

        # Создаём событие
        event = PlateEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            camera_id=self.camera_id,
            track_id=track_id,
            frame_idx=frame_idx,
            # Три score
            car_score=scores["car_score"],
            plate_score=scores["plate_score"],
            ocr_score=scores["ocr_score"],
            total_score=scores["total"],
            # Legacy
            detection_conf=detection_conf,
            plate_conf=plate_conf,
            ocr_conf=scores["total"],  # для совместимости
            processing_time_ms=round(processing_time, 1),
            plate_text=text,
            region=region,
            # Raw данные
            brightness=round(brightness, 1),
            blur=round(blur, 1),
            plate_width_px=plate_width,
            plate_height_px=plate_height,
            car_width_px=car_width,
            car_height_px=car_height,
            crop=car_crop,
        )

        # === ФИЛЬТР 3: Длина текста ===
        if len(text) < self.min_plate_chars:
            self.stats["skipped_chars"] += 1
            event.reject_reason = f"chars:{len(text)}"
            self._update_failed(track_id, event)
            return self._get_best_result(track_id)

        # === ФИЛЬТР 4: Формат номера ===
        if self.plate_format_regex:
            if not re.match(self.plate_format_regex, text):
                self.stats["skipped_format"] += 1
                event.reject_reason = f"format:{text}"
                self._update_failed(track_id, event)
                return self._get_best_result(track_id)

        # Прошёл все фильтры!
        self.stats["passed"] += 1
        self._update_passed(track_id, event)

        return self.passed_results.get(track_id)

    def _update_passed(self, track_id: int, event: PlateEvent):
        """Обновляет лучший passed результат"""
        current = self.passed_results.get(track_id)
        if current is None or event.total_score > current.total_score:
            # Копируем кроп только когда сохраняем лучший результат
            if event.crop is not None:
                event.crop = event.crop.copy()
            self.passed_results[track_id] = event
            print(f"✓ {event.plate_text} | car:{event.car_score:.2f} plate:{event.plate_score:.2f} ocr:{event.ocr_score:.2f}")

            # Логируем в файл
            if self.file_logger:
                self.file_logger.log_ocr_attempt(
                    frame_idx=event.frame_idx,
                    track_id=track_id,
                    status="passed",
                    plate_text=event.plate_text,
                    car_score=event.car_score,
                    plate_score=event.plate_score,
                    ocr_score=event.ocr_score,
                    blur=event.blur,
                    brightness=event.brightness,
                    plate_width=event.plate_width_px,
                    plate_height=event.plate_height_px,
                    car_width=event.car_width_px,
                    car_height=event.car_height_px,
                    processing_ms=event.processing_time_ms,
                )

    def _update_failed(self, track_id: int, event: PlateEvent):
        """Обновляет лучший failed результат"""
        current = self.failed_results.get(track_id)
        if current is None or event.total_score > current.total_score:
            # Копируем кроп только когда сохраняем лучший результат
            if event.crop is not None:
                event.crop = event.crop.copy()
            self.failed_results[track_id] = event

            # Логируем в файл
            if self.file_logger:
                self.file_logger.log_ocr_attempt(
                    frame_idx=event.frame_idx,
                    track_id=track_id,
                    status="failed",
                    plate_text=event.plate_text,
                    car_score=event.car_score,
                    plate_score=event.plate_score,
                    ocr_score=event.ocr_score,
                    blur=event.blur,
                    brightness=event.brightness,
                    plate_width=event.plate_width_px,
                    plate_height=event.plate_height_px,
                    car_width=event.car_width_px,
                    car_height=event.car_height_px,
                    reason=event.reject_reason,
                    processing_ms=event.processing_time_ms,
                )

    def _get_best_result(self, track_id: int) -> Optional[PlateEvent]:
        """Возвращает лучший результат (сначала passed, потом failed)"""
        if track_id in self.passed_results:
            return self.passed_results[track_id]
        return self.failed_results.get(track_id)

    def get_result(self, track_id: int) -> Optional[PlateEvent]:
        return self._get_best_result(track_id)

    def finalize(self):
        """Сохраняет все результаты на диск."""
        print(f"\nSaving results...")
        print(f"   Total frames processed: {self.stats['total_frames']}")
        print(f"   Skipped (car size): {self.stats['skipped_car_size']}")
        print(f"   Skipped (quality): {self.stats['skipped_quality']}")
        print(f"   Skipped (not better): {self.stats['skipped_not_better']}")
        print(f"   Skipped (cooldown): {self.stats['skipped_cooldown']}")
        print(f"   OCR called: {self.stats['ocr_called']}")
        print(f"   OCR no plate: {self.stats['ocr_no_plate']}")
        print(f"   Skipped (plate size): {self.stats['skipped_plate_size']}")
        print(f"   Skipped (few chars): {self.stats['skipped_chars']}")
        print(f"   Skipped (format): {self.stats['skipped_format']}")

        # Сохраняем PASSED
        passed_data = []
        for track_id, event in self.passed_results.items():
            filename = f"{event.event_id}_{event.plate_text}.jpg"
            filepath = os.path.join(self.passed_images, filename)

            if event.crop is not None:
                cv2.imwrite(filepath, event.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            event.crop_path = f"images/{filename}"
            passed_data.append(event.to_dict())
            print(f"   ✓ {event.plate_text} | car:{event.car_score:.2f} plate:{event.plate_score:.2f} ocr:{event.ocr_score:.2f} = {event.total_score:.2f}")

        with open(os.path.join(self.passed_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(passed_data, f, ensure_ascii=False, indent=2)

        # Сохраняем FAILED
        failed_data = []
        for track_id, event in self.failed_results.items():
            # Не сохраняем failed если есть passed для этого track_id
            if track_id in self.passed_results:
                continue

            filename = f"{event.event_id}_{event.plate_text}.jpg"
            filepath = os.path.join(self.failed_images, filename)

            if event.crop is not None:
                cv2.imwrite(filepath, event.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            event.crop_path = f"images/{filename}"
            failed_data.append(event.to_dict())

        with open(os.path.join(self.failed_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(failed_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults:")
        print(f"   PASSED: {len(passed_data)} -> {self.passed_dir}")
        print(f"   FAILED: {len(failed_data)} -> {self.failed_dir}")

        # Детальное профилирование OCR
        self.print_ocr_profile()

    def get_stats(self) -> dict:
        return self.stats

    def get_ocr_profile(self) -> dict:
        """Возвращает детальное профилирование OCR из pipeline"""
        if hasattr(self.pipeline, 'get_profile'):
            return self.pipeline.get_profile()
        return {}

    def print_ocr_profile(self):
        """Выводит детальное профилирование OCR"""
        prof = self.get_ocr_profile()
        if not prof:
            return
        print(f"\n📊 Pipeline профилирование ({prof['count']} вызовов):")
        print(f"   localization (YOLO): {prof['localization_ms']:6.2f} мс")
        print(f"   ocr (text read):     {prof['ocr_ms']:6.2f} мс")
        print(f"   ─────────────────────────────")
        print(f"   ИТОГО:               {prof['total_ms']:6.2f} мс/вызов")
