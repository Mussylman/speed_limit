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
from collections import defaultdict
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

from kz_plate import (
    fix_kz_plate, fix_kz_8chars, kz_score, kz_score_7,
    merge_texts_charwise, levenshtein_distance,
    CHAR_TO_DIGIT, CHAR_TO_LETTER, LETTER_CONFUSIONS, OCR_PREFER, KZ_REGIONS,
)


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
    ocr_conf: float = 0.0        # реальная уверенность в тексте (длина + формат + структура)
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
        min_blur_score: float = 5.0,
        min_brightness: float = 20.0,
        max_brightness: float = 240.0,
        quality_improvement: float = 1.0,
        min_confirmations: int = 2,
        no_plate_max: int = 3,
        no_plate_cooldown_sec: float = 0.3,
        video_start_time=None,
        video_fps: float = 25.0,
    ):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.video_start_time = video_start_time
        self.video_fps = video_fps
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

        # Папка для немедленного сохранения хороших результатов (переживает Ctrl+C)
        self.res_ocr_dir = os.path.join(output_dir, "res_ocr")
        os.makedirs(self.res_ocr_dir, exist_ok=True)

        # NomeroffNet с классификацией (нужна для OCR confidence)
        print("Loading NomeroffNet...")
        self.pipeline = pipeline(
            "number_plate_detection_and_reading_runtime",
            off_number_plate_classification=False,  # ON — даёт реальный ocr_conf
            default_label="kz",                     # Казахстан
            default_lines_count=1,
            path_to_model=os.path.join(os.path.dirname(os.path.dirname(__file__)), "nomeroff-net", "data", "models", "Detector", "yolov11x", "yolov11x-keypoints-2024-10-11.engine"),
        )
        print("NomeroffNet ready (yolov11x-keypoints TensorRT, classification ON, default=kz)")

        # Для 1080p кропы машин ≤960px — resize не нужен
        self.max_ocr_width = 0  # 0 = no resize

        # Лучшие результаты
        self.passed_results: Dict[int, PlateEvent] = {}  # прошли все фильтры
        self.failed_results: Dict[int, PlateEvent] = {}  # не прошли формат, но лучшие

        # Двухуровневая оптимизация
        self.best_quality: Dict[int, float] = {}  # лучший quality score по track_id
        self.last_ocr_frame: Dict[int, int] = {}  # cooldown

        # Пороги качества для дешёвого отбора
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.quality_improvement = quality_improvement
        self.min_confirmations = min_confirmations

        # Мульти-кадровое голосование
        self.text_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pending_results: Dict[int, PlateEvent] = {}

        # Full frame buffer (JPEG bytes, keyed by frame_idx, set from main.py)
        self.full_frames: Dict[int, np.ndarray] = {}  # frame_idx -> cv2.imencode buffer

        # No-plate cooldown: после N неудачных OCR (номер не найден) —
        # пауза на X секунд, чтобы не тратить GPU
        self.no_plate_max = no_plate_max
        self.no_plate_cooldown_frames = int(no_plate_cooldown_sec * video_fps)
        self.no_plate_count: Dict[int, int] = {}        # track_id -> consecutive no_plate count
        self.no_plate_until: Dict[int, int] = {}        # track_id -> frame_idx when cooldown ends

        # Per-position character history: all chars seen at each position across all reads
        # Used to correct confusable letters (e.g., Z read as A on most scales)
        self.char_history: Dict[int, Dict[int, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )  # track_id -> position -> {char: count}

        # Статистика
        self.stats = {
            "total_frames": 0,
            "skipped_car_size": 0,
            "skipped_quality": 0,
            "skipped_not_better": 0,
            "skipped_cooldown": 0,
            "skipped_no_plate_cooldown": 0,
            "ocr_called": 0,
            "ocr_no_plate": 0,
            "skipped_plate_size": 0,
            "skipped_chars": 0,
            "skipped_format": 0,
            "skipped_low_conf": 0,
            "unconfirmed": 0,
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

    def _fuzzy_vote(self, track_id: int, text: str) -> Tuple[str, int]:
        """Fuzzy голосование: тексты с расстоянием ≤1 считаются одинаковыми.

        Returns: (best_text, total_votes)
        """
        votes = self.text_votes[track_id]
        # Найти существующий текст с расстоянием ≤ 1
        for existing_text in votes:
            if levenshtein_distance(text, existing_text) <= 1:
                # Голосуем за тот вариант, который длиннее или чаще
                if votes[existing_text] >= votes.get(text, 0):
                    votes[existing_text] += 1
                    return existing_text, votes[existing_text]
                else:
                    # Новый вариант лучше — переносим голоса
                    votes[text] = votes.pop(existing_text) + 1
                    return text, votes[text]
        # Новый текст
        votes[text] += 1
        return text, votes[text]

    def _compute_ocr_confidence(self, text: str) -> float:
        """Реальная уверенность в тексте OCR на основе длины, формата и структуры."""
        if not text:
            return 0.0
        score = 0.0
        # Длина: ожидаем min_plate_chars (8 для KZ)
        if len(text) >= self.min_plate_chars:
            score += 0.4
        elif len(text) >= self.min_plate_chars - 1:
            score += 0.2
        # Формат regex
        if self.plate_format_regex:
            if re.match(self.plate_format_regex, text):
                score += 0.4
        else:
            score += 0.2  # нет regex = нейтрально
        # Структура символов (KZ: 3 цифры + 3 буквы + 2 цифры)
        if len(text) >= 8:
            d_ok = all(c.isdigit() for c in text[:3]) and all(c.isdigit() for c in text[6:8])
            l_ok = all(c.isalpha() for c in text[3:6])
            if d_ok and l_ok:
                score += 0.2
        return min(1.0, score)

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
        # Blur (Laplacian variance): < 10000 = размыто (0.3), > 20000 = чётко (1.0)
        blur_norm = min(1.0, max(0.3, (blur - 10000) / 15000 * 0.7 + 0.3))

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

    # KZ plate utilities imported from kz_plate module

    def _run_pipeline_once(self, car_crop: np.ndarray, scale: float = 1.0,
                           return_bbox: bool = False
                           ) -> Optional[Tuple]:
        """Run NomeroffNet on a single crop.

        Returns (text, plate_conf, ocr_conf, pw, ph, region) or None.
        If return_bbox=True, appends (x1, y1, x2, y2) pixel coords in car_crop.
        """
        rgb = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
        res = self.pipeline([rgb])

        if isinstance(res, (list, tuple)) and len(res) > 0:
            data = res[0]
        else:
            return None

        if not isinstance(data, (list, tuple)) or len(data) < 9:
            return None

        bboxs = data[1]
        texts = data[8]
        ocr_confs = data[7]
        regions = data[5] if len(data) > 5 else []

        if not bboxs or len(bboxs) == 0:
            return None

        bbox = bboxs[0]
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        plate_width = int((x2 - x1) / scale)
        plate_height = int((y2 - y1) / scale)

        plate_conf = float(bbox[4]) if len(bbox) > 4 else 0.5

        if plate_width < self.min_plate_width or plate_height < self.min_plate_height:
            return None

        text = texts[0] if texts and len(texts) > 0 else ""
        text = text.replace(" ", "").upper()
        text = fix_kz_plate(text)

        region = regions[0] if regions and len(regions) > 0 else "unknown"

        ocr_conf = 0.0
        if ocr_confs and len(ocr_confs) > 0:
            raw = ocr_confs[0]
            if isinstance(raw, (list, tuple, np.ndarray)):
                vals = [float(v) for v in raw if float(v) >= 0]
                ocr_conf = min(vals) if vals else 0.0
            elif isinstance(raw, (int, float)) and float(raw) >= 0:
                ocr_conf = float(raw)

        result = (text, plate_conf, ocr_conf, plate_width, plate_height, region)
        if return_bbox:
            result = result + (int(x1), int(y1), int(x2), int(y2))
        return result

    def _reocr_plate_zoom(self, car_crop: np.ndarray, target_plate_width: int = 400
                          ) -> Optional[Tuple[str, float, float, int, int, str]]:
        """Detect plate bbox, extract & zoom plate region, re-run OCR.

        This gives NomeroffNet a much higher resolution view of the plate
        characters, helping distinguish confusable letters like Z vs A.
        """
        # Step 1: detect plate bbox at original scale
        r = self._run_pipeline_once(car_crop, 1.0, return_bbox=True)
        if r is None or len(r) < 10:
            return None

        text0, plate_conf, ocr_conf0, pw, ph, region, bx1, by1, bx2, by2 = r
        h, w = car_crop.shape[:2]

        # Step 2: extract plate region with padding
        pad_x = int((bx2 - bx1) * 0.3)
        pad_y = int((by2 - by1) * 1.0)  # vertical padding for context
        rx1 = max(0, bx1 - pad_x)
        ry1 = max(0, by1 - pad_y)
        rx2 = min(w, bx2 + pad_x)
        ry2 = min(h, by2 + pad_y)

        plate_region = car_crop[ry1:ry2, rx1:rx2]
        if plate_region.size == 0:
            return None

        # Step 3: upscale plate region to target width
        pr_h, pr_w = plate_region.shape[:2]
        if pr_w < 30:
            return None
        zoom = target_plate_width / pr_w
        zoomed = cv2.resize(plate_region, None, fx=zoom, fy=zoom,
                            interpolation=cv2.INTER_LANCZOS4)

        # Step 4: re-run OCR on zoomed plate
        r2 = self._run_pipeline_once(zoomed, zoom)
        if r2 is None:
            return None

        text2 = r2[0]
        text2 = fix_kz_plate(text2)
        return (text2, plate_conf, r2[2], pw, ph, region)

    def _correct_from_history(self, track_id: int, text: str) -> str:
        """Use char_history to correct confusable letters.

        Uses _OCR_PREFER table: if at a letter position we've seen both
        char X and char Y, and (X,Y) has a known preferred direction,
        pick the preferred one — but only if the alternative was seen
        at least 2 times (to filter single-read noise).

        When multiple alternatives match, picks the one with the
        highest vote count.
        """
        if len(text) != 8 or track_id not in self.char_history:
            return text

        chars = list(text)
        history = self.char_history[track_id]
        changed = False

        # Only correct letter positions (3, 4, 5)
        for i in [3, 4, 5]:
            if i not in history:
                continue
            current = chars[i]
            if not current.isalpha():
                continue

            # Collect all valid preferred alternatives with their vote counts
            best_alt = None
            best_alt_count = 0

            for alt_char, alt_count in history[i].items():
                if alt_char == current or not alt_char.isalpha():
                    continue
                # Must have been seen at least 2 times to filter noise
                if alt_count < 2:
                    continue
                # Check _OCR_PREFER for this pair (both directions)
                preferred = OCR_PREFER.get((current, alt_char))
                if preferred is None:
                    preferred = OCR_PREFER.get((alt_char, current))
                if preferred and preferred != current and alt_count > best_alt_count:
                    best_alt = preferred
                    best_alt_count = alt_count

            if best_alt:
                test_chars = chars.copy()
                test_chars[i] = best_alt
                test_text = ''.join(test_chars)
                if self.plate_format_regex and re.match(self.plate_format_regex, test_text):
                    chars[i] = best_alt
                    changed = True

        result = ''.join(chars)
        if changed and result != text:
            print(f"  [char_history] {text} -> {result} (cross-read correction)")
        return result

    def _detect_plate(self, car_crop: np.ndarray) -> Optional[Tuple[str, float, float, int, int, str]]:
        """
        Распознаёт номер через NomeroffNet.

        For small crops (width < 640), runs OCR at multiple scales
        and merges results via per-character majority voting.

        Returns: (text, plate_conf, ocr_conf, plate_width, plate_height, region) или None
        """
        if car_crop is None or car_crop.size == 0:
            return None

        try:
            h, w = car_crop.shape[:2]

            # Downscale large crops if limit is set
            if self.max_ocr_width > 0 and w > self.max_ocr_width:
                scale = self.max_ocr_width / w
                crop = cv2.resize(car_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                return self._run_pipeline_once(crop, scale)

            # For small crops: multi-scale OCR with character-level voting
            if w < 960:
                results = []
                # Scale 1: original
                r = self._run_pipeline_once(car_crop, 1.0)
                if r:
                    results.append(r)
                # Scale 2: upscale to 480px (skip if already bigger)
                if w < 480:
                    s2 = 480 / w
                    crop2 = cv2.resize(car_crop, None, fx=s2, fy=s2, interpolation=cv2.INTER_LANCZOS4)
                    r = self._run_pipeline_once(crop2, s2)
                    if r:
                        results.append(r)
                # Scale 3: upscale to 640px (skip if already bigger)
                if w < 640:
                    s3 = 640 / w
                    crop3 = cv2.resize(car_crop, None, fx=s3, fy=s3, interpolation=cv2.INTER_LANCZOS4)
                    r = self._run_pipeline_once(crop3, s3)
                    if r:
                        results.append(r)
                # Scale 4: upscale to 800px
                if w < 800:
                    s4 = 800 / w
                    crop4 = cv2.resize(car_crop, None, fx=s4, fy=s4, interpolation=cv2.INTER_LANCZOS4)
                    r = self._run_pipeline_once(crop4, s4)
                    if r:
                        results.append(r)
                # Scale 5: upscale to 960px
                if w < 960:
                    s5 = 960 / w
                    crop5 = cv2.resize(car_crop, None, fx=s5, fy=s5, interpolation=cv2.INTER_LANCZOS4)
                    r = self._run_pipeline_once(crop5, s5)
                    if r:
                        results.append(r)

                # Plate-zoom: detect plate bbox, extract & zoom, re-OCR
                zoom_r = self._reocr_plate_zoom(car_crop, target_plate_width=400)
                if zoom_r:
                    results.append(zoom_r)

                if not results:
                    self.stats["skipped_plate_size"] += 1
                    return None

                # Single result — return as-is
                if len(results) == 1:
                    return results[0]

                # Multiple results: merge texts that are 8 chars (KZ format)
                kz_results = [r for r in results if len(r[0]) == 8]
                if len(kz_results) >= 2:
                    texts = [r[0] for r in kz_results]
                    merged_text = merge_texts_charwise(texts)
                    merged_text = fix_kz_plate(merged_text)
                    # Use metadata from highest plate_conf result
                    best_r = max(kz_results, key=lambda r: r[1])
                    return (merged_text, best_r[1], best_r[2], best_r[3], best_r[4], best_r[5])

                # Fallback: return highest plate_conf result
                return max(results, key=lambda r: r[1])

            # Normal size crop: also try plate-zoom
            results = []
            r = self._run_pipeline_once(car_crop, 1.0)
            if r:
                results.append(r)
            zoom_r = self._reocr_plate_zoom(car_crop, target_plate_width=400)
            if zoom_r:
                results.append(zoom_r)
            if not results:
                return None
            if len(results) == 1:
                return results[0]
            # Merge if both are 8 chars
            kz_results = [r for r in results if len(r[0]) == 8]
            if len(kz_results) >= 2:
                texts = [r[0] for r in kz_results]
                merged = merge_texts_charwise(texts)
                merged = fix_kz_plate(merged)
                best_r = max(kz_results, key=lambda r: r[1])
                return (merged, best_r[1], best_r[2], best_r[3], best_r[4], best_r[5])
            return max(results, key=lambda r: r[1])

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

        # Cleanup old full frames (keep last 200)
        if len(self.full_frames) > 200:
            cutoff = sorted(self.full_frames.keys())[-200]
            self.full_frames = {k: v for k, v in self.full_frames.items() if k >= cutoff}

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

        # Staleness: если трек не видели 100+ кадров, track_id скорее всего
        # переиспользован для другой машины — сбрасываем историю
        if last_frame >= 0 and frame_idx - last_frame > 100:
            self.char_history.pop(track_id, None)
            self.best_quality.pop(track_id, None)
            self.text_votes.pop(track_id, None)
            self.pending_results.pop(track_id, None)
            self.no_plate_count.pop(track_id, None)
            self.no_plate_until.pop(track_id, None)

        # === ФИЛЬТР 5: No-plate cooldown ===
        # Если номер не найден N раз подряд — пауза
        if track_id in self.no_plate_until:
            if frame_idx < self.no_plate_until[track_id]:
                self.stats["skipped_no_plate_cooldown"] += 1
                return self._get_best_result(track_id)
            else:
                # Cooldown истёк — сбрасываем
                del self.no_plate_until[track_id]
                self.no_plate_count.pop(track_id, None)

        # Кадр — претендент! Запускаем дорогой pipeline
        self.last_ocr_frame[track_id] = frame_idx
        self.stats["ocr_called"] += 1

        start_time = time.time()
        plate_info = self._detect_plate(car_crop)
        processing_time = (time.time() - start_time) * 1000

        if plate_info is None:
            self.stats["ocr_no_plate"] += 1
            # Считаем consecutive no_plate для cooldown
            cnt = self.no_plate_count.get(track_id, 0) + 1
            self.no_plate_count[track_id] = cnt
            if cnt >= self.no_plate_max:
                self.no_plate_until[track_id] = frame_idx + self.no_plate_cooldown_frames
            # НЕ обновляем best_quality — даём шанс следующим кадрам
            return self._get_best_result(track_id)

        # Номер найден — сбрасываем no_plate счётчик и поднимаем планку качества
        self.no_plate_count.pop(track_id, None)
        self.best_quality[track_id] = quality

        text, plate_conf, nomeroff_ocr_conf, plate_width, plate_height, region = plate_info

        # Record per-position chars for cross-read correction
        # Only from structurally plausible KZ readings (filter garbage)
        if len(text) == 8 and kz_score(text) >= 5:
            for i, ch in enumerate(text):
                self.char_history[track_id][i][ch] += 1
        elif len(text) == 7 and kz_score_7(text) >= 4:
            for i, ch in enumerate(text):
                self.char_history[track_id][i][ch] += 1

        # OCR confidence: реальный из NomeroffNet + бонус за формат
        text_conf = self._compute_ocr_confidence(text)
        if nomeroff_ocr_conf > 0:
            # Комбинируем: NomeroffNet confidence (70%) + текстовый анализ (30%)
            ocr_conf = nomeroff_ocr_conf * 0.7 + text_conf * 0.3
        else:
            ocr_conf = text_conf

        # Проверяем формат для ocr_score (используется в total_score для выбора лучшего кропа)
        format_valid = True
        if self.plate_format_regex:
            format_valid = bool(re.match(self.plate_format_regex, text))

        # Вычисляем три отдельных score (total_score — метрика качества кадра)
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
        if self.video_start_time is not None and frame_idx >= 0:
            from datetime import timedelta
            evt_ts = (self.video_start_time + timedelta(seconds=frame_idx / self.video_fps)).isoformat()
        else:
            evt_ts = datetime.now().isoformat()
        event = PlateEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=evt_ts,
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
            ocr_conf=ocr_conf,  # реальная уверенность в тексте
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

        # === ФИЛЬТР: OCR confidence ===
        if ocr_conf < self.min_conf:
            self.stats["skipped_low_conf"] += 1
            event.reject_reason = f"low_conf:{ocr_conf:.2f}"
            self._update_failed(track_id, event)
            return self._get_best_result(track_id)

        # === ФИЛЬТР: Длина текста ===
        if len(text) < self.min_plate_chars:
            self.stats["skipped_chars"] += 1
            event.reject_reason = f"chars:{len(text)}"
            self._update_failed(track_id, event)
            return self._get_best_result(track_id)

        # === ФИЛЬТР: Формат номера ===
        if self.plate_format_regex:
            if not re.match(self.plate_format_regex, text):
                self.stats["skipped_format"] += 1
                event.reject_reason = f"format:{text}"
                self._update_failed(track_id, event)
                return self._get_best_result(track_id)

        # === Мульти-кадровое голосование (fuzzy: 1 символ допуск) ===
        best_text, votes = self._fuzzy_vote(track_id, text)
        if best_text != text:
            # Fuzzy match скорректировал текст
            text = best_text
            event.plate_text = text

        if votes < self.min_confirmations:
            self.stats["unconfirmed"] += 1
            # Сохраняем лучший pending (по total_score для выбора кропа)
            current_pending = self.pending_results.get(track_id)
            if current_pending is None or event.total_score > current_pending.total_score:
                if event.crop is not None:
                    event.crop = event.crop.copy()
                self.pending_results[track_id] = event
            return self._get_best_result(track_id)

        # Прошёл все фильтры + подтверждён голосованием!
        self.stats["passed"] += 1

        # Cross-read correction: use rare chars from history (e.g., Z seen once vs A seen many times)
        corrected_text = self._correct_from_history(track_id, text)
        if corrected_text != text:
            text = corrected_text
            event.plate_text = text

        # Используем лучший кроп из pending если он лучше
        pending = self.pending_results.pop(track_id, None)
        if pending and pending.total_score > event.total_score:
            pending.plate_text = text  # apply corrected text
            pending.ocr_conf = ocr_conf
            self._update_passed(track_id, pending)
        else:
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

            # Сразу сохраняем на диск (переживает Ctrl+C)
            self._save_to_res_ocr(event)

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

    def _save_to_res_ocr(self, event: PlateEvent):
        """Немедленно сохраняет кроп, полный кадр и JSON на диск (переживает Ctrl+C)"""
        try:
            ts = event.timestamp.replace(":", "-").replace(".", "-")
            basename = f"{ts}_{event.plate_text}_id{event.track_id}"

            if event.crop is not None:
                img_path = os.path.join(self.res_ocr_dir, f"{basename}.jpg")
                cv2.imwrite(img_path, event.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save full frame (JPEG buffer from main.py)
            full_jpg = self.full_frames.get(event.frame_idx)
            if full_jpg is not None:
                full_path = os.path.join(self.res_ocr_dir, f"{basename}_full.jpg")
                with open(full_path, 'wb') as f:
                    f.write(full_jpg.tobytes())

            json_path = os.path.join(self.res_ocr_dir, f"{basename}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(event.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"res_ocr save error: {e}")

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
        """Возвращает лучший результат (passed > pending > failed)"""
        if track_id in self.passed_results:
            return self.passed_results[track_id]
        if track_id in self.pending_results:
            return self.pending_results[track_id]
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
        print(f"   Skipped (no-plate cooldown): {self.stats['skipped_no_plate_cooldown']}")
        print(f"   OCR called: {self.stats['ocr_called']}")
        print(f"   OCR no plate: {self.stats['ocr_no_plate']}")
        print(f"   Skipped (plate size): {self.stats['skipped_plate_size']}")
        print(f"   Skipped (few chars): {self.stats['skipped_chars']}")
        print(f"   Skipped (format): {self.stats['skipped_format']}")
        print(f"   Skipped (low conf): {self.stats['skipped_low_conf']}")
        print(f"   Unconfirmed (votes): {self.stats['unconfirmed']}")

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
