# async_ocr.py
# Асинхронный OCR в отдельном потоке
# Оптимизации: дедуп, upgrade (вместо early stop), TTL кэша

import cv2
import re
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
    orig_crop: np.ndarray = None  # original hi-res crop (before 800px resize)


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
    car_width: int = 0  # ширина кропа (для upgrade score)


class AsyncOCR:
    """
    Асинхронный OCR в отдельном потоке.

    Логика Upgrade (вместо слепого Early Stop):
    1. Очередь с ограничением + drop policy
    2. Дедуп по track_id (не больше 1 задачи в очереди на track)
    3. Upgrade: если новый кроп крупнее — пропускаем на OCR, сравниваем score
       Score = ocr_conf * width_factor — приоритет крупным кропам
       Максимум max_upgrades апгрейдов на один трек
    4. TTL/очистка кэша по track_id
    """

    def __init__(
        self,
        recognizer,
        max_queue_size: int = 64,
        num_workers: int = 1,
        max_crop_width: int = 640,
        good_conf_threshold: float = 0.88,
        cache_max_size: int = 200,
        cache_ttl_frames: int = 300,  # ~10 сек при 30 fps
        batch_size: int = 4,
        batch_timeout_ms: float = 50.0,
        max_upgrades: int = 3,
    ):
        self.recognizer = recognizer
        self.max_queue_size = max_queue_size
        self.max_crop_width = max_crop_width
        self.good_conf_threshold = good_conf_threshold
        self.cache_max_size = cache_max_size
        self.cache_ttl_frames = cache_ttl_frames
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self.max_upgrades = max_upgrades

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
            "dropped_full": 0,         # очередь полна
            "dropped_in_flight": 0,    # уже в очереди
            "dropped_max_upgrades": 0, # лимит апгрейдов исчерпан
            "dropped_not_bigger": 0,   # кроп не крупнее предыдущего
            "dropped_stable": 0,       # Double Check: текст подтверждён
            "upgrades": 0,             # успешные апгрейды (score улучшился)
            "total_queue_time_ms": 0.0,
            "total_processing_time_ms": 0.0,
        }

        # Воркеры
        self.running = True
        self.workers = []
        for i in range(num_workers):
            t = Thread(target=self._worker_batch, name=f"OCR-Worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

        # Кэш: {track_id: (result, frame_idx, score, ocr_count, max_width)}
        #   result     — лучший PlateEvent (или None)
        #   frame_idx  — кадр лучшего результата (для TTL)
        #   score      — лучший score (conf * width_factor)
        #   ocr_count  — сколько раз OCR обработал этот track
        #   max_width  — максимальная ширина кропа, которую уже отправляли
        self.results_cache: Dict[int, tuple] = {}
        self.cache_lock = Lock()

        # Text frequency voting: most-seen text wins (not highest score)
        # {track_id: {text: (count, best_result, best_score, frame_idx, car_width)}}
        self.text_votes: Dict[int, Dict[str, tuple]] = {}

    def _vote_weight(self, text: str, ocr_conf: float) -> float:
        """Вес голоса: валидный KZ-формат и высокий conf весят больше мусора."""
        w = 1.0
        # Текст проходит формат-регулярку → +2.0
        fmt_regex = getattr(self.recognizer, 'plate_format_regex', '')
        if fmt_regex and text and re.match(fmt_regex, text):
            w += 2.0
        # Высокая уверенность OCR → +1.5
        if ocr_conf > 0.9:
            w += 1.5
        return w

    @staticmethod
    def _calc_score(result, car_width: int) -> float:
        """Качество результата: OCR conf + plate quality, взвешенный размером.

        score = (ocr_conf * 0.6 + plate_score * 0.4) * width_factor
        Штраф x0.5 если текст не проходит полную валидацию (text_conf < 1.0).
        """
        if result is None or not hasattr(result, 'ocr_conf'):
            return 0.0
        conf = result.ocr_conf
        plate_score = getattr(result, 'plate_score', 0.5)
        width_factor = min(car_width / 500.0, 1.0)
        score = (conf * 0.6 + plate_score * 0.4) * width_factor
        # Штраф за несоответствие формату
        text_conf = getattr(result, 'text_conf', 1.0)
        if text_conf < 1.0:
            score *= 0.5
        return score

    def _worker_batch(self):
        """OCR batch worker: собирает до batch_size задач, обрабатывает пачкой."""
        while self.running:
            # Собираем пачку задач
            batch = []
            try:
                task = self.input_queue.get(timeout=0.1)
                if task is None:
                    break
                batch.append(task)
            except Empty:
                continue

            # Пытаемся набрать ещё (без долгого ожидания)
            deadline = time.time() + self.batch_timeout_s
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    task = self.input_queue.get_nowait()
                    if task is None:
                        break
                    batch.append(task)
                except Empty:
                    break

            # Обрабатываем всю пачку
            for task in batch:
                queue_time_ms = (time.time() - task.submit_time) * 1000

                t_start = time.time()
                try:
                    result = self.recognizer.process(
                        track_id=task.track_id,
                        car_crop=task.crop,
                        car_height=task.car_height,
                        car_width=task.car_width,
                        frame_idx=task.frame_idx,
                        detection_conf=task.detection_conf,
                        orig_crop=task.orig_crop,
                    )
                except Exception as e:
                    print(f"\n[AsyncOCR] Error: {e}")
                    result = None

                processing_time_ms = (time.time() - t_start) * 1000

                blur = getattr(self.recognizer, 'last_blur', 0.0)
                brightness = getattr(self.recognizer, 'last_brightness', 0.0)

                ocr_result = OCRResult(
                    track_id=task.track_id,
                    frame_idx=task.frame_idx,
                    result=result,
                    processing_time_ms=processing_time_ms,
                    queue_time_ms=queue_time_ms,
                    blur_score=blur,
                    brightness_score=brightness,
                    car_width=task.car_width,
                )

                self.output_queue.put(ocr_result)

                with self.in_flight_lock:
                    self.in_flight.discard(task.track_id)

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
        orig_crop: np.ndarray = None,
    ) -> bool:
        """
        Отправить кроп на OCR (неблокирующий).

        Returns:
            True если задача добавлена, False если пропущена
        """
        # [0] Stability disabled: early wrong readings can lock in bad text
        # Rely on max_upgrades to limit OCR calls instead
        # if track_id in self.stable_tracks:
        #     self.stats["dropped_stable"] += 1
        #     return False

        # [1] Очередь полна → drop
        if self.input_queue.full():
            self.stats["dropped_full"] += 1
            return False

        # [2] Дедуп: уже в очереди → skip
        with self.in_flight_lock:
            if track_id in self.in_flight:
                self.stats["dropped_in_flight"] += 1
                return False

        # [3] Upgrade logic: пропускаем если кроп крупнее предыдущего
        #     и лимит апгрейдов не исчерпан
        with self.cache_lock:
            cached = self.results_cache.get(track_id)
            if cached:
                _, _, _, ocr_count, max_width = cached
                if ocr_count >= self.max_upgrades + 1:
                    # Лимит: 1 начальный + max_upgrades апгрейдов
                    self.stats["dropped_max_upgrades"] += 1
                    return False
                if car_width < max_width * 0.9:
                    # Кроп значительно меньше — не может улучшить результат
                    # Допуск 90%: OCR качество зависит не только от ширины
                    self.stats["dropped_not_bigger"] += 1
                    return False

        # Добавляем в in_flight
        with self.in_flight_lock:
            self.in_flight.add(track_id)

        task = OCRTask(
            track_id=track_id,
            crop=crop.copy(),
            car_height=car_height,
            car_width=car_width,
            frame_idx=frame_idx,
            detection_conf=detection_conf,
            submit_time=time.time(),
            orig_crop=orig_crop.copy() if orig_crop is not None else None,
        )

        self.input_queue.put(task)
        self.stats["submitted"] += 1
        return True

    def get_results(self, current_frame_idx: int = 0) -> list:
        """
        Забрать все готовые результаты (неблокирующий).

        Text frequency voting: most-seen text wins.
        For each track, we count how many times each text was returned by OCR.
        The winning text is the one with the highest count (ties broken by score).
        This is more robust than score-based selection because NomeroffNet reads
        the correct plate on the majority of frames.

        Returns:
            Список OCRResult
        """
        results = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                results.append(result)

                new_score = self._calc_score(result.result, result.car_width)
                new_text = ""
                if result.result and hasattr(result.result, 'plate_text'):
                    new_text = result.result.plate_text

                tid = result.track_id

                with self.cache_lock:
                    cached = self.results_cache.get(tid)
                    # Only count format-valid results toward max_upgrades budget
                    # so garbage reads from small crops don't waste upgrade slots
                    fmt_regex = getattr(self.recognizer, 'plate_format_regex', '')
                    is_valid = bool(fmt_regex and new_text and re.match(fmt_regex, new_text))
                    if cached:
                        _, _, _, ocr_count, max_width = cached
                        new_max_width = max(max_width, result.car_width)
                        new_count = ocr_count + (1 if is_valid else 0)
                    else:
                        new_max_width = result.car_width
                        new_count = 1 if is_valid else 0

                    # Update text weighted votes
                    if new_text:
                        if tid not in self.text_votes:
                            self.text_votes[tid] = {}
                        votes = self.text_votes[tid]
                        ocr_conf = getattr(result.result, 'ocr_conf', 0.0)
                        vote_w = self._vote_weight(new_text, ocr_conf)
                        if new_text in votes:
                            old_wsum, old_res, old_sc, old_fi, old_cw = votes[new_text]
                            # Keep best result for this text variant
                            if new_score > old_sc:
                                votes[new_text] = (old_wsum + vote_w, result.result,
                                                   new_score, result.frame_idx,
                                                   result.car_width)
                            else:
                                votes[new_text] = (old_wsum + vote_w, old_res,
                                                   old_sc, old_fi, old_cw)
                        else:
                            votes[new_text] = (vote_w, result.result, new_score,
                                               result.frame_idx, result.car_width)

                        # Pick winner: highest weighted sum, ties broken by score
                        best_text = max(
                            votes.keys(),
                            key=lambda t: (votes[t][0], votes[t][2]),
                        )
                        winner = votes[best_text]
                        w_wsum, w_result, w_score, w_frame, w_cw = winner

                        # Check if winner changed
                        old_winner_text = ""
                        if cached and cached[0]:
                            old_winner_text = getattr(cached[0], 'plate_text', '')
                        if old_winner_text and best_text != old_winner_text:
                            print(f"  [vote] t{tid}: {old_winner_text} -> "
                                  f"{best_text} (w={w_wsum:.1f})")

                        self.results_cache[tid] = (
                            w_result, w_frame, w_score,
                            new_count, new_max_width,
                        )
                        if new_count > 1:
                            self.stats["upgrades"] += 1
                    else:
                        # No text (None result) — just update counters
                        if cached:
                            old_result, old_frame, old_score, _, _ = cached
                            self.results_cache[tid] = (
                                old_result, old_frame, old_score,
                                new_count, new_max_width,
                            )
                        else:
                            self.results_cache[tid] = (
                                result.result, result.frame_idx,
                                new_score, 1, result.car_width,
                            )
            except Empty:
                break

        # Очистка старых записей по TTL
        if current_frame_idx > 0 and len(self.results_cache) > 0:
            self._cleanup_cache(current_frame_idx)

        return results

    def _cleanup_cache(self, current_frame_idx: int):
        """Очистка кэша по TTL и размеру"""
        with self.cache_lock:
            # Удаляем по TTL (frame_idx — индекс 1 в кортеже)
            expired = [
                tid for tid, entry in self.results_cache.items()
                if current_frame_idx - entry[1] > self.cache_ttl_frames
            ]
            for tid in expired:
                del self.results_cache[tid]
                self.text_votes.pop(tid, None)

            # Если всё ещё слишком много — удаляем самые старые
            if len(self.results_cache) > self.cache_max_size:
                sorted_items = sorted(
                    self.results_cache.items(),
                    key=lambda x: x[1][1]  # по frame_idx
                )
                to_remove = len(self.results_cache) - self.cache_max_size
                for tid, _ in sorted_items[:to_remove]:
                    del self.results_cache[tid]
                    self.text_votes.pop(tid, None)

    def sync_to_recognizer(self, recognizer, final: bool = False):
        """Sync weighted voting winners back to plate_recognizer.passed_results.

        When async_ocr's voting picks a different text than what
        plate_recognizer stored, update plate_recognizer so that saved results
        and video overlays reflect the voting winner.

        Vote-promote: if a track never reached min_confirmations in the
        recognizer's internal _fuzzy_vote (varied OCR outputs), but has
        enough weighted votes here, promote it to passed_results.

        Protection: never replace a valid-KZ plate with an invalid one.
        """
        fmt_regex = getattr(recognizer, 'plate_format_regex', '')
        with self.cache_lock:
            for tid, votes in self.text_votes.items():
                if not votes:
                    continue

                if final:
                    # Print vote summary for debugging
                    sorted_texts = sorted(votes.items(),
                                          key=lambda x: (-x[1][0], -x[1][2]))
                    summary = ", ".join(f"{t}:w{v[0]:.1f}" for t, v in sorted_texts[:5])
                    print(f"  [votes] t{tid}: {summary}")

                best_text = max(
                    votes.keys(),
                    key=lambda t: (votes[t][0], votes[t][2]),
                )
                wsum = votes[best_text][0]
                if wsum < 3.0:
                    continue  # need meaningful weight to override
                if tid in recognizer.passed_results:
                    existing = recognizer.passed_results[tid]
                    if existing.plate_text != best_text:
                        # Protection: don't replace valid KZ with another valid KZ
                        # The recognizer's internal confirmation is more reliable
                        if fmt_regex:
                            old_valid = bool(re.match(fmt_regex, existing.plate_text))
                            new_valid = bool(re.match(fmt_regex, best_text))
                            if old_valid:
                                # Confirmed valid plate — don't override
                                if final:
                                    print(f"  [vote-sync] t{tid}: KEEP "
                                          f"{existing.plate_text}(confirmed) over {best_text}(w={wsum:.1f})")
                                continue
                            if not new_valid:
                                if final:
                                    print(f"  [vote-sync] t{tid}: BLOCKED "
                                          f"{existing.plate_text} -> {best_text}(invalid)")
                                continue
                        print(f"  [vote-sync] t{tid}: "
                              f"{existing.plate_text} -> {best_text} (w={wsum:.1f})")
                        existing.plate_text = best_text
                elif wsum >= 6.0:
                    # Vote-promote: track never passed min_confirmations
                    # but has enough weighted votes (>= 2 format-valid reads)
                    if fmt_regex and not re.match(fmt_regex, best_text):
                        continue  # only promote format-valid texts
                    # Use the best PlateEvent stored for this text variant
                    w_result = votes[best_text][1]
                    if w_result is None:
                        # Try pending_results as fallback
                        w_result = recognizer.pending_results.get(tid)
                    if w_result is None:
                        continue
                    w_result.plate_text = best_text
                    recognizer.stats["passed"] += 1
                    recognizer._update_passed(tid, w_result)
                    recognizer.pending_results.pop(tid, None)
                    if final:
                        print(f"  [vote-promote] t{tid}: {best_text} "
                              f"(w={wsum:.1f}) -> PASSED")

    def get_cached_result(self, track_id: int) -> Optional[Any]:
        """Получить лучший закэшированный результат для track_id"""
        with self.cache_lock:
            cached = self.results_cache.get(track_id)
            if cached:
                return cached[0]  # (result, frame_idx, score, count, width) → result
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
        stats["dropped_total"] = (stats["dropped_full"] + stats["dropped_in_flight"]
                                  + stats["dropped_max_upgrades"] + stats["dropped_not_bigger"])
        return stats

    def print_stats(self):
        """Вывод статистики"""
        stats = self.get_stats()
        print(f"\nAsyncOCR Stats:")
        print(f"   Submitted:     {stats['submitted']:6d}")
        print(f"   Processed:     {stats['processed']:6d}")
        print(f"   Upgrades:      {stats['upgrades']:6d}")
        print(f"   Dropped total: {stats['dropped_total']:6d}")
        print(f"     - queue full:    {stats['dropped_full']:6d}")
        print(f"     - in flight:     {stats['dropped_in_flight']:6d}")
        print(f"     - max upgrades:  {stats['dropped_max_upgrades']:6d}")
        print(f"     - not bigger:    {stats['dropped_not_bigger']:6d}")
        print(f"   Avg queue:     {stats['avg_queue_time_ms']:6.1f} ms")
        print(f"   Avg process:   {stats['avg_processing_time_ms']:6.1f} ms")
        print(f"   Cache size:    {stats['cache_size']:6d}")
