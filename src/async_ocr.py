# async_ocr.py
# Асинхронный OCR в отдельном потоке
# Оптимизации: дедуп, early stop, resize, TTL кэша

import cv2
import time
from threading import Thread, Lock
from queue import Queue, Empty
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class OCRTask:
    """Задача для OCR"""
    track_id: int
    crop: np.ndarray  # уже уменьшенный!
    car_height: int
    car_width: int
    frame_idx: int
    detection_conf: float
    submit_time: float = 0.0


@dataclass
class OCRResult:
    """Результат OCR"""
    track_id: int
    frame_idx: int
    result: Any  # PlateEvent или None
    processing_time_ms: float
    queue_time_ms: float  # время в очереди
    blur_score: float = 0.0
    brightness_score: float = 0.0


class AsyncOCR:
    """
    Асинхронный OCR в отдельном потоке.

    Оптимизации:
    1. Очередь с ограничением + drop policy
    2. Дедуп по track_id (не больше 1 задачи в очереди на track)
    3. Early stop по качеству OCR (conf >= threshold → не submit)
    4. Resize crop перед отправкой (экономия RAM/CPU)
    5. TTL/очистка кэша по track_id
    """

    def __init__(
        self,
        recognizer,
        max_queue_size: int = 64,
        num_workers: int = 1,
        max_crop_width: int = 320,
        good_conf_threshold: float = 0.88,
        cache_max_size: int = 200,
        cache_ttl_frames: int = 300,  # ~10 сек при 30 fps
    ):
        self.recognizer = recognizer
        self.max_queue_size = max_queue_size
        self.max_crop_width = max_crop_width
        self.good_conf_threshold = good_conf_threshold
        self.cache_max_size = cache_max_size
        self.cache_ttl_frames = cache_ttl_frames

        # Очереди
        self.input_queue: Queue[OCRTask] = Queue(maxsize=max_queue_size)
        self.output_queue: Queue[OCRResult] = Queue()

        # [2] Дедуп: track_id которые сейчас в очереди/обработке
        self.in_flight: Set[int] = set()
        self.in_flight_lock = Lock()

        # Статистика
        self.stats = {
            "submitted": 0,
            "processed": 0,
            "dropped_full": 0,      # очередь полна
            "dropped_in_flight": 0, # уже в очереди
            "dropped_good_conf": 0, # уже хороший результат
            "total_queue_time_ms": 0.0,
            "total_processing_time_ms": 0.0,
        }

        # Воркеры
        self.running = True
        self.workers = []
        for i in range(num_workers):
            t = Thread(target=self._worker, name=f"OCR-Worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

        # [5] Кэш с TTL: {track_id: (result, last_frame_idx)}
        self.results_cache: Dict[int, tuple] = {}
        self.cache_lock = Lock()

    def _worker(self):
        """OCR воркер в отдельном потоке"""
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            if task is None:  # stop signal
                break

            # Время в очереди
            queue_time_ms = (time.time() - task.submit_time) * 1000

            # OCR
            t_start = time.time()
            try:
                result = self.recognizer.process(
                    track_id=task.track_id,
                    car_crop=task.crop,
                    car_height=task.car_height,
                    car_width=task.car_width,
                    frame_idx=task.frame_idx,
                    detection_conf=task.detection_conf,
                )
            except Exception as e:
                print(f"\n[AsyncOCR] Error: {e}")
                result = None

            processing_time_ms = (time.time() - t_start) * 1000

            # Получаем quality метрики из recognizer
            blur = getattr(self.recognizer, 'last_blur', 0.0)
            brightness = getattr(self.recognizer, 'last_brightness', 0.0)

            # Результат
            ocr_result = OCRResult(
                track_id=task.track_id,
                frame_idx=task.frame_idx,
                result=result,
                processing_time_ms=processing_time_ms,
                queue_time_ms=queue_time_ms,
                blur_score=blur,
                brightness_score=brightness,
            )

            self.output_queue.put(ocr_result)

            # [2] Убираем из in_flight
            with self.in_flight_lock:
                self.in_flight.discard(task.track_id)

            # Статистика
            self.stats["processed"] += 1
            self.stats["total_queue_time_ms"] += queue_time_ms
            self.stats["total_processing_time_ms"] += processing_time_ms

    def submit(
        self,
        track_id: int,
        crop: np.ndarray,
        car_height: int = 0,
        car_width: int = 0,
        frame_idx: int = 0,
        detection_conf: float = 0.0,
    ) -> bool:
        """
        Отправить кроп на OCR (неблокирующий).

        Returns:
            True если задача добавлена, False если пропущена
        """
        # [1] Очередь полна → drop
        if self.input_queue.full():
            self.stats["dropped_full"] += 1
            return False

        # [2] Дедуп: уже в очереди → skip
        with self.in_flight_lock:
            if track_id in self.in_flight:
                self.stats["dropped_in_flight"] += 1
                return False

        # [3] Early stop: уже хороший результат → skip
        with self.cache_lock:
            cached = self.results_cache.get(track_id)
            if cached:
                result, _ = cached
                if result and hasattr(result, 'ocr_conf') and result.ocr_conf >= self.good_conf_threshold:
                    self.stats["dropped_good_conf"] += 1
                    return False

        # [4] Resize crop для экономии памяти
        # НЕ масштабируем car_height/car_width - они нужны оригинальные для фильтров!
        h, w = crop.shape[:2]
        if w > self.max_crop_width:
            scale = self.max_crop_width / w
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Добавляем в in_flight
        with self.in_flight_lock:
            self.in_flight.add(track_id)

        task = OCRTask(
            track_id=track_id,
            crop=crop.copy(),  # копируем уменьшенный crop
            car_height=car_height,
            car_width=car_width,
            frame_idx=frame_idx,
            detection_conf=detection_conf,
            submit_time=time.time(),
        )

        self.input_queue.put(task)
        self.stats["submitted"] += 1
        return True

    def get_results(self, current_frame_idx: int = 0) -> list:
        """
        Забрать все готовые результаты (неблокирующий).

        Args:
            current_frame_idx: текущий кадр для TTL

        Returns:
            Список OCRResult
        """
        results = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                results.append(result)
                # [5] Обновляем кэш с frame_idx для TTL
                with self.cache_lock:
                    if result.result is not None:
                        self.results_cache[result.track_id] = (result.result, result.frame_idx)
            except Empty:
                break

        # [5] Очистка старых записей по TTL
        if current_frame_idx > 0 and len(self.results_cache) > 0:
            self._cleanup_cache(current_frame_idx)

        return results

    def _cleanup_cache(self, current_frame_idx: int):
        """[5] Очистка кэша по TTL и размеру"""
        with self.cache_lock:
            # Удаляем по TTL
            expired = [
                tid for tid, (_, frame_idx) in self.results_cache.items()
                if current_frame_idx - frame_idx > self.cache_ttl_frames
            ]
            for tid in expired:
                del self.results_cache[tid]

            # Если всё ещё слишком много — удаляем самые старые
            if len(self.results_cache) > self.cache_max_size:
                sorted_items = sorted(
                    self.results_cache.items(),
                    key=lambda x: x[1][1]  # по frame_idx
                )
                to_remove = len(self.results_cache) - self.cache_max_size
                for tid, _ in sorted_items[:to_remove]:
                    del self.results_cache[tid]

    def get_cached_result(self, track_id: int) -> Optional[Any]:
        """Получить закэшированный результат для track_id"""
        with self.cache_lock:
            cached = self.results_cache.get(track_id)
            if cached:
                return cached[0]  # (result, frame_idx) -> result
            return None

    def queue_size(self) -> int:
        """Текущий размер очереди"""
        return self.input_queue.qsize()

    def pending_results(self) -> int:
        """Количество готовых результатов в очереди"""
        return self.output_queue.qsize()

    def stop(self):
        """Остановить воркеры"""
        self.running = False
        # Отправляем stop signal
        for _ in self.workers:
            try:
                self.input_queue.put(None, timeout=0.1)
            except:
                pass
        # Ждём завершения
        for t in self.workers:
            t.join(timeout=2.0)

    def get_stats(self) -> dict:
        """Статистика работы"""
        stats = self.stats.copy()
        if stats["processed"] > 0:
            stats["avg_queue_time_ms"] = stats["total_queue_time_ms"] / stats["processed"]
            stats["avg_processing_time_ms"] = stats["total_processing_time_ms"] / stats["processed"]
        else:
            stats["avg_queue_time_ms"] = 0
            stats["avg_processing_time_ms"] = 0
        stats["queue_size"] = self.queue_size()
        with self.cache_lock:
            stats["cache_size"] = len(self.results_cache)
        with self.in_flight_lock:
            stats["in_flight"] = len(self.in_flight)
        stats["dropped_total"] = stats["dropped_full"] + stats["dropped_in_flight"] + stats["dropped_good_conf"]
        return stats

    def print_stats(self):
        """Вывод статистики"""
        stats = self.get_stats()
        print(f"\n📊 AsyncOCR Stats:")
        print(f"   Submitted:     {stats['submitted']:6d}")
        print(f"   Processed:     {stats['processed']:6d}")
        print(f"   Dropped total: {stats['dropped_total']:6d}")
        print(f"     - queue full:  {stats['dropped_full']:6d}")
        print(f"     - in flight:   {stats['dropped_in_flight']:6d}")
        print(f"     - good conf:   {stats['dropped_good_conf']:6d}")
        print(f"   Avg queue:     {stats['avg_queue_time_ms']:6.1f} ms")
        print(f"   Avg process:   {stats['avg_processing_time_ms']:6.1f} ms")
        print(f"   Cache size:    {stats['cache_size']:6d}")
