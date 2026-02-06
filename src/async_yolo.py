"""
Асинхронный YOLO в отдельном потоке.

Видео показывается с полным FPS, детекции накладываются когда готовы.
"""

import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np


@dataclass
class YOLOTask:
    """Задача для YOLO."""
    frame: np.ndarray
    frame_idx: int
    submit_time: float


@dataclass
class DetectionData:
    """Данные одной детекции."""
    obj_id: int
    box: tuple  # (x1, y1, x2, y2) int
    conf: float
    crop: np.ndarray  # Вырезанный crop машины для OCR
    cx: int  # Центр X
    cy: int  # Низ Y (для скорости)


@dataclass
class YOLOResult:
    """Результат YOLO."""
    frame_idx: int
    detections: List[DetectionData]  # Список детекций с crops
    frame_shape: tuple  # (h, w) для отрисовки
    processing_time_ms: float
    queue_time_ms: float


class AsyncYOLO:
    """
    Асинхронный YOLO в отдельном потоке.

    Архитектура:
    - Main loop отправляет кадры в очередь (не блокируется)
    - Worker thread обрабатывает YOLO
    - Main loop забирает готовые результаты
    """

    def __init__(
        self,
        model,
        imgsz: int = 1280,
        classes: List[int] = None,
        half: bool = True,
        tracker: str = "bytetrack.yaml",
        max_queue_size: int = 3,
    ):
        self.model = model
        self.imgsz = imgsz
        self.classes = classes or [2]  # cars by default
        self.half = half
        self.tracker = tracker

        # Очередь кадров (size=3 чтобы не терять кадры с хорошими номерами)
        self.input_queue: Queue[YOLOTask] = Queue(maxsize=max_queue_size)
        self.output_queue: Queue[YOLOResult] = Queue()

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        # Статистика
        self.stats = {
            "submitted": 0,
            "processed": 0,
            "dropped": 0,
        }

    def _worker(self):
        """YOLO worker в отдельном потоке."""
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            if task is None:  # stop signal
                break

            queue_time_ms = (time.time() - task.submit_time) * 1000

            # YOLO inference
            t_start = time.time()
            frame = task.frame
            h, w = frame.shape[:2]

            try:
                results = self.model.track(
                    frame,
                    imgsz=self.imgsz,
                    classes=self.classes,
                    persist=True,
                    verbose=False,
                    half=self.half,
                    tracker=self.tracker,
                )
                processing_time_ms = (time.time() - t_start) * 1000

                # Извлекаем детекции и вырезаем crops
                detections = []

                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes_obj = results[0].boxes
                    xyxy = boxes_obj.xyxy.cpu().numpy()
                    ids = boxes_obj.id.int().cpu().numpy()
                    confs = boxes_obj.conf.cpu().numpy()

                    for i in range(len(ids)):
                        x1, y1, x2, y2 = xyxy[i]
                        obj_id = int(ids[i])
                        conf = float(confs[i])

                        if conf < 0.5:
                            continue

                        # Clip to frame bounds
                        x1i, y1i = max(0, int(x1)), max(0, int(y1))
                        x2i, y2i = min(w, int(x2)), min(h, int(y2))

                        if x2i <= x1i or y2i <= y1i:
                            continue

                        # Вырезаем crop (копия, т.к. frame будет переиспользован)
                        crop = frame[y1i:y2i, x1i:x2i].copy()

                        if crop.size == 0:
                            continue

                        # Центр для скорости
                        cx = int((x1 + x2) / 2)
                        cy = int(y2)

                        detections.append(DetectionData(
                            obj_id=obj_id,
                            box=(x1i, y1i, x2i, y2i),
                            conf=conf,
                            crop=crop,
                            cx=cx,
                            cy=cy,
                        ))

                result = YOLOResult(
                    frame_idx=task.frame_idx,
                    detections=detections,
                    frame_shape=(h, w),
                    processing_time_ms=processing_time_ms,
                    queue_time_ms=queue_time_ms,
                )

            except Exception as e:
                print(f"\n⚠️ AsyncYOLO error: {e}")
                result = YOLOResult(
                    frame_idx=task.frame_idx,
                    detections=[],
                    frame_shape=(h, w),
                    processing_time_ms=(time.time() - t_start) * 1000,
                    queue_time_ms=queue_time_ms,
                )

            self.output_queue.put(result)
            self.stats["processed"] += 1

    def submit(self, frame: np.ndarray, frame_idx: int) -> bool:
        """
        Отправить кадр на обработку.
        Returns: True если добавлен, False если очередь полна.
        """
        if self.input_queue.full():
            self.stats["dropped"] += 1
            return False

        task = YOLOTask(
            frame=frame.copy(),
            frame_idx=frame_idx,
            submit_time=time.time(),
        )

        try:
            self.input_queue.put_nowait(task)
            self.stats["submitted"] += 1
            return True
        except:
            self.stats["dropped"] += 1
            return False

    def get_results(self) -> List[YOLOResult]:
        """Забрать все готовые результаты (неблокирующий)."""
        results = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                results.append(result)
            except Empty:
                break
        return results

    def get_latest_result(self) -> Optional[YOLOResult]:
        """Забрать только последний результат (остальные отбрасываются)."""
        latest = None
        while True:
            try:
                result = self.output_queue.get_nowait()
                latest = result
            except Empty:
                break
        return latest

    def stop(self):
        """Остановить worker."""
        self.running = False
        try:
            self.input_queue.put_nowait(None)  # stop signal
        except:
            pass
        self.worker_thread.join(timeout=2.0)

    def get_stats(self) -> dict:
        """Получить статистику."""
        return {
            **self.stats,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
        }
