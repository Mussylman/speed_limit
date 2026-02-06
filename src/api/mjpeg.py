# mjpeg.py
# MJPEG стриминг для превью

import cv2
import asyncio
import time
import numpy as np
from typing import Dict, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field
from threading import Thread, Lock, Event
from queue import Queue, Empty

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from video.decoder import HWDecoder


@dataclass
class StreamStats:
    frames_read: int = 0
    frames_served: int = 0
    reconnects: int = 0
    fps: float = 0.0
    backend: str = "None"


@dataclass
class StreamSession:
    camera_id: str
    url: str
    running: bool = False
    decoder: Optional[HWDecoder] = None
    thread: Optional[Thread] = None
    frame_queue: Queue = field(default_factory=lambda: Queue(maxsize=3))
    last_frame: Optional[np.ndarray] = None
    frame_lock: Lock = field(default_factory=Lock)
    stop_event: Event = field(default_factory=Event)
    stats: StreamStats = field(default_factory=StreamStats)
    frame_processor: Optional[Callable] = None


class MJPEGStreamer:
    """
    MJPEG стример для HTTP preview.

    Функции:
    - Запуск/остановка потоков
    - MJPEG generator для StreamingResponse
    - Snapshot (текущий кадр)
    - Auto-reconnect
    """

    def __init__(
        self,
        jpeg_quality: int = 75,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
    ):
        self.jpeg_quality = jpeg_quality
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects

        self._streams: Dict[str, StreamSession] = {}
        self._lock = Lock()

    def start(
        self,
        camera_id: str,
        url: str,
        frame_processor: Optional[Callable] = None,
    ) -> bool:
        """Запускает стрим"""
        with self._lock:
            if camera_id in self._streams:
                if self._streams[camera_id].running:
                    return True

            session = StreamSession(
                camera_id=camera_id,
                url=url,
                frame_processor=frame_processor,
            )
            self._streams[camera_id] = session

        session.running = True
        session.stop_event.clear()

        thread = Thread(
            target=self._stream_loop,
            args=(session,),
            daemon=True,
        )
        session.thread = thread
        thread.start()

        return True

    def stop(self, camera_id: str) -> bool:
        """Останавливает стрим"""
        with self._lock:
            session = self._streams.get(camera_id)
            if not session:
                return False

        session.running = False
        session.stop_event.set()

        if session.thread:
            session.thread.join(timeout=3.0)

        if session.decoder:
            session.decoder.close()

        with self._lock:
            del self._streams[camera_id]

        return True

    def _stream_loop(self, session: StreamSession):
        """Основной цикл чтения"""
        reconnects = 0

        while session.running and reconnects < self.max_reconnects:
            session.decoder = HWDecoder(prefer_hw=True, verbose=False)
            info = session.decoder.open(session.url)

            if info.cap is None:
                reconnects += 1
                time.sleep(self.reconnect_delay)
                continue

            session.stats.backend = info.backend.value
            reconnects = 0

            fps_counter = 0
            fps_start = time.time()

            while session.running and not session.stop_event.is_set():
                ok, frame = session.decoder.read()

                if not ok or frame is None:
                    break

                session.stats.frames_read += 1

                # Process frame
                if session.frame_processor:
                    try:
                        frame = session.frame_processor(frame)
                    except Exception:
                        pass

                # Save last frame
                with session.frame_lock:
                    session.last_frame = frame

                # Queue for MJPEG
                if session.frame_queue.full():
                    try:
                        session.frame_queue.get_nowait()
                    except Empty:
                        pass
                session.frame_queue.put(frame)

                # FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    session.stats.fps = fps_counter
                    fps_counter = 0
                    fps_start = time.time()

            session.decoder.close()
            session.stats.reconnects += 1
            reconnects += 1

            if session.running:
                time.sleep(self.reconnect_delay)

    async def generate(self, camera_id: str) -> AsyncGenerator[bytes, None]:
        """MJPEG generator"""
        session = self._streams.get(camera_id)
        if not session:
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        while session.running:
            try:
                frame = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.frame_queue.get(timeout=1.0)
                )
            except Empty:
                continue

            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer.tobytes()

            session.stats.frames_served += 1

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

    def get_snapshot(self, camera_id: str) -> Optional[bytes]:
        """Текущий кадр как JPEG"""
        session = self._streams.get(camera_id)
        if not session:
            return None

        with session.frame_lock:
            if session.last_frame is None:
                return None
            frame = session.last_frame.copy()

        _, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return buffer.tobytes()

    def get_status(self, camera_id: str) -> Optional[dict]:
        """Статус стрима"""
        session = self._streams.get(camera_id)
        if not session:
            return None

        return {
            "camera_id": camera_id,
            "running": session.running,
            "backend": session.stats.backend,
            "fps": session.stats.fps,
            "frames_read": session.stats.frames_read,
            "frames_served": session.stats.frames_served,
            "reconnects": session.stats.reconnects,
        }

    def list_streams(self) -> list:
        """Список активных стримов"""
        with self._lock:
            return list(self._streams.keys())

    def stop_all(self):
        """Останавливает все"""
        with self._lock:
            ids = list(self._streams.keys())
        for cam_id in ids:
            self.stop(cam_id)


# Глобальный экземпляр
mjpeg_streamer = MJPEGStreamer()
