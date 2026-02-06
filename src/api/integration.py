# integration.py
# Интеграция pipeline → API (WebSocket события)

import asyncio
from typing import Optional, Dict, Any
from threading import Thread
from datetime import datetime

from .websocket import ws_manager


class PipelineEvents:
    """
    Отправка событий из detection pipeline в API.

    Использование в main.py:
        from api.integration import pipeline_events

        # При обновлении скорости
        pipeline_events.speed_update(track_id=1, speed_kmh=85.3, camera_id="cam1")

        # При нарушении
        pipeline_events.violation(track_id=1, plate="123ABC01", speed=95, limit=70)

        # При распознавании номера
        pipeline_events.plate_recognized(track_id=1, plate="123ABC01", confidence=0.95)
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._started = False

    def start(self):
        """Запускает async loop в отдельном потоке"""
        if self._started:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True, name="PipelineEvents")
        self._thread.start()
        self._started = True

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _emit(self, channel: str, event: str, data: Any):
        """Отправляет событие в WebSocket"""
        if self._loop is None:
            return

        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(channel, event, data),
            self._loop
        )

    # =========================================================
    # События
    # =========================================================

    def speed_update(
        self,
        track_id: int,
        speed_kmh: float,
        camera_id: str,
        **extra,
    ):
        """Обновление скорости"""
        data = {
            "track_id": track_id,
            "speed_kmh": round(speed_kmh, 1),
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        self._emit(f"speed:{camera_id}", "update", data)

    def violation(
        self,
        track_id: int,
        plate: str,
        speed_kmh: float,
        limit_kmh: float,
        camera_id: str,
        image_path: Optional[str] = None,
        **extra,
    ):
        """Нарушение скоростного режима"""
        data = {
            "track_id": track_id,
            "plate": plate,
            "speed_kmh": round(speed_kmh, 1),
            "limit_kmh": limit_kmh,
            "over_speed": round(speed_kmh - limit_kmh, 1),
            "camera_id": camera_id,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        self._emit(f"violations:{camera_id}", "new", data)
        self._emit("violations", "new", data)  # Общий канал

    def plate_recognized(
        self,
        track_id: int,
        plate: str,
        confidence: float,
        camera_id: str,
        speed_kmh: Optional[float] = None,
        **extra,
    ):
        """Распознан номер"""
        data = {
            "track_id": track_id,
            "plate": plate,
            "confidence": round(confidence, 3),
            "speed_kmh": round(speed_kmh, 1) if speed_kmh else None,
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        self._emit(f"plates:{camera_id}", "recognized", data)

    def stats(
        self,
        frame_idx: int,
        detections: int,
        fps: float,
        camera_id: str,
        **extra,
    ):
        """Статистика (раз в секунду)"""
        data = {
            "frame_idx": frame_idx,
            "detections": detections,
            "fps": round(fps, 1),
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        self._emit(f"stats:{camera_id}", "update", data)

    def system_event(self, event: str, data: Any):
        """Системное событие"""
        self._emit("system", event, data)

    def stop(self):
        """Останавливает"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=2.0)
            self._loop = None
            self._thread = None
            self._started = False


# Глобальный экземпляр
pipeline_events = PipelineEvents()
