# metrics_logger.py
# Сбор и вывод метрик производительности и качества

import os
import json
import time
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime, timedelta


@dataclass
class FrameMetrics:
    """Метрики одного кадра"""
    frame_idx: int = 0
    timestamp: float = 0.0

    # Timing (ms)
    yolo_ms: float = 0.0
    ocr_ms: float = 0.0
    draw_ms: float = 0.0
    total_ms: float = 0.0

    # Detections
    cars_count: int = 0
    cars_conf_min: float = 0.0
    cars_conf_max: float = 0.0
    cars_conf_avg: float = 0.0

    # OCR
    ocr_called: bool = False
    ocr_skipped_reason: str = ""
    ocr_result: str = ""
    ocr_conf: float = 0.0

    # Quality
    blur_score: float = 0.0
    brightness_score: float = 0.0

    # Speed
    speeds: List[float] = field(default_factory=list)
    max_speed: float = 0.0


class MetricsLogger:
    """
    Логгер метрик с live-выводом и JSON сохранением.

    Функции:
    - Live-строка каждый кадр
    - Подробный отчёт каждые N секунд
    - JSON лог в файл
    """

    def __init__(
        self,
        output_dir: str,
        report_interval: float = 5.0,  # секунд между отчётами
        json_log: bool = True,
        live_line: bool = False,  # Отключено - логируется в файлы
    ):
        self.output_dir = output_dir
        self.report_interval = report_interval
        self.json_log = json_log
        self.live_line = live_line

        # Файл для JSON
        self.json_path = os.path.join(output_dir, "metrics.jsonl")
        self.json_file = None
        if json_log:
            self.json_file = open(self.json_path, "w", encoding="utf-8")

        # Текущий кадр
        self.current = FrameMetrics()

        # Аккумуляторы для статистики
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.total_frames = 0

        # Скользящие окна для средних (последние 100 кадров)
        self.window_size = 100
        self.yolo_times: deque = deque(maxlen=self.window_size)
        self.ocr_times: deque = deque(maxlen=self.window_size)
        self.total_times: deque = deque(maxlen=self.window_size)
        self.cars_counts: deque = deque(maxlen=self.window_size)
        self.fps_window: deque = deque(maxlen=30)  # для FPS

        # Quality метрики
        self.blur_scores: deque = deque(maxlen=self.window_size)
        self.brightness_scores: deque = deque(maxlen=self.window_size)

        # Накопительная статистика
        self.total_cars = 0
        self.total_ocr_calls = 0
        self.total_ocr_passed = 0
        self.total_ocr_failed = 0
        self.total_violations = 0
        self.all_speeds: List[float] = []

        # OCR skip reasons
        self.skip_reasons: Dict[str, int] = {
            "car_size": 0,
            "quality": 0,
            "not_better": 0,
            "cooldown": 0,
            "no_plate": 0,
            "plate_size": 0,
            "chars": 0,
            "format": 0,
        }

        # Для FPS расчёта
        self.fps_frame_times: deque = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # Ссылка на async_ocr для отображения очереди
        self.async_ocr = None

    def start_frame(self, frame_idx: int):
        """Начало обработки кадра"""
        self.current = FrameMetrics(
            frame_idx=frame_idx,
            timestamp=time.time() - self.start_time,
        )
        self._frame_start = time.time()

    def log_yolo(self, time_ms: float, cars_count: int, confidences: List[float]):
        """Логирует результаты YOLO"""
        self.current.yolo_ms = time_ms
        self.current.cars_count = cars_count
        self.total_cars += cars_count

        if confidences:
            self.current.cars_conf_min = min(confidences)
            self.current.cars_conf_max = max(confidences)
            self.current.cars_conf_avg = sum(confidences) / len(confidences)

        self.yolo_times.append(time_ms)
        self.cars_counts.append(cars_count)

    def log_ocr(self, called: bool, time_ms: float = 0, result: str = "",
                conf: float = 0, passed: bool = False, skip_reason: str = ""):
        """Логирует результаты OCR"""
        if called:
            self.current.ocr_called = True
            self.current.ocr_ms = time_ms
            self.current.ocr_result = result
            self.current.ocr_conf = conf
            self.total_ocr_calls += 1
            self.ocr_times.append(time_ms)

            if passed:
                self.total_ocr_passed += 1
            elif result:  # был результат но не прошёл фильтры
                self.total_ocr_failed += 1
        else:
            self.current.ocr_skipped_reason = skip_reason
            if skip_reason in self.skip_reasons:
                self.skip_reasons[skip_reason] += 1

    def log_quality(self, blur: float, brightness: float):
        """Логирует метрики качества (берём лучший blur за кадр)"""
        # Сохраняем лучший blur (больше = резче = лучше)
        if blur > self.current.blur_score:
            self.current.blur_score = blur
            self.current.brightness_score = brightness  # brightness от лучшего кропа

        # Для статистики
        if blur > 0:
            self.blur_scores.append(blur)
        if brightness > 0:
            self.brightness_scores.append(brightness)

    def log_speed(self, speed: float, is_violation: bool = False):
        """Логирует скорость"""
        self.current.speeds.append(speed)
        if speed > self.current.max_speed:
            self.current.max_speed = speed
        self.all_speeds.append(speed)
        if is_violation:
            self.total_violations += 1

    def log_draw(self, time_ms: float):
        """Логирует время отрисовки"""
        self.current.draw_ms = time_ms

    def end_frame(self):
        """Завершение обработки кадра"""
        self.current.total_ms = (time.time() - self._frame_start) * 1000
        self.total_times.append(self.current.total_ms)
        self.total_frames += 1

        # FPS
        now = time.time()
        self.fps_frame_times.append(now)
        if len(self.fps_frame_times) >= 2:
            dt = self.fps_frame_times[-1] - self.fps_frame_times[0]
            if dt > 0:
                self.current_fps = (len(self.fps_frame_times) - 1) / dt

        # JSON лог
        if self.json_file:
            self.json_file.write(json.dumps(asdict(self.current), ensure_ascii=False) + "\n")

        # Live line
        if self.live_line:
            self._print_live_line()

        # Периодический отчёт
        if now - self.last_report_time >= self.report_interval:
            self._print_report()
            self.last_report_time = now

    def _print_live_line(self):
        """Выводит live-строку (обновляется на месте)"""
        elapsed = timedelta(seconds=int(time.time() - self.start_time))

        # Формат: [HH:MM:SS] Frame XXXX | FPS: XX.X | Cars: X | OCR: X/X | Speed: XX km/h
        ocr_status = f"{self.current.ocr_result}" if self.current.ocr_called else "-"
        speed_str = f"{self.current.max_speed:.0f}" if self.current.max_speed > 0 else "-"

        # OCR queue info
        ocr_queue = ""
        if self.async_ocr:
            q = self.async_ocr.queue_size()
            if q > 0:
                ocr_queue = f" Q:{q}"

        line = (
            f"\r[{elapsed}] "
            f"Frame {self.current.frame_idx:5d} | "
            f"FPS: {self.current_fps:5.1f} | "
            f"Cars: {self.current.cars_count} | "
            f"YOLO: {self.current.yolo_ms:5.1f}ms | "
            f"OCR: {ocr_status:12s}{ocr_queue} | "
            f"Speed: {speed_str:>3s} km/h"
        )

        # Очищаем до конца строки и выводим
        sys.stdout.write(line + " " * 10)
        sys.stdout.flush()

    def _print_report(self):
        """Выводит компактный отчёт в 2 строки"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Средние значения
        avg_yolo = sum(self.yolo_times) / len(self.yolo_times) if self.yolo_times else 0
        avg_ocr = sum(self.ocr_times) / len(self.ocr_times) if self.ocr_times else 0
        avg_cars = sum(self.cars_counts) / len(self.cars_counts) if self.cars_counts else 0

        avg_speed = sum(self.all_speeds) / len(self.all_speeds) if self.all_speeds else 0
        max_speed = max(self.all_speeds) if self.all_speeds else 0

        ocr_rate = (self.total_ocr_passed / self.total_ocr_calls * 100) if self.total_ocr_calls > 0 else 0

        avg_blur = sum(self.blur_scores) / len(self.blur_scores) if self.blur_scores else 0
        avg_bright = sum(self.brightness_scores) / len(self.brightness_scores) if self.brightness_scores else 0

        # Компактный вывод в 2 строки
        print(f"\n[{elapsed_str}] FPS:{self.current_fps:5.1f} | YOLO:{avg_yolo:5.1f}ms | OCR:{self.total_ocr_passed}/{self.total_ocr_calls}({ocr_rate:.0f}%) | Cars:{self.total_cars} ({avg_cars:.1f}/f) | Speed:{avg_speed:.0f}/{max_speed:.0f} km/h | Viol:{self.total_violations}")
        print(f"         Blur:{avg_blur:.0f} Bright:{avg_bright:.0f} | Skip: sz:{self.skip_reasons['car_size']} q:{self.skip_reasons['quality']} cd:{self.skip_reasons['cooldown']} nop:{self.skip_reasons['no_plate']}")

    def update_skip_reasons(self, stats: Dict[str, int]):
        """Обновляет счётчики skip reasons из PlateRecognizer"""
        mapping = {
            "skipped_car_size": "car_size",
            "skipped_quality": "quality",
            "skipped_not_better": "not_better",
            "skipped_cooldown": "cooldown",
            "ocr_no_plate": "no_plate",
            "skipped_plate_size": "plate_size",
            "skipped_chars": "chars",
            "skipped_format": "format",
        }
        for src, dst in mapping.items():
            if src in stats:
                self.skip_reasons[dst] = stats[src]

    def finalize(self):
        """Финальный отчёт и закрытие файлов"""
        # Последний отчёт
        self._print_report()

        # Закрываем JSON
        if self.json_file:
            self.json_file.close()
            print(f"📄 Metrics saved: {self.json_path}")

        # Сохраняем сводку
        summary = {
            "total_frames": self.total_frames,
            "duration_sec": time.time() - self.start_time,
            "avg_fps": self.current_fps,
            "total_cars": self.total_cars,
            "total_ocr_calls": self.total_ocr_calls,
            "total_ocr_passed": self.total_ocr_passed,
            "total_violations": self.total_violations,
            "avg_speed": sum(self.all_speeds) / len(self.all_speeds) if self.all_speeds else 0,
            "max_speed": max(self.all_speeds) if self.all_speeds else 0,
            "skip_reasons": self.skip_reasons,
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📄 Summary saved: {summary_path}")
