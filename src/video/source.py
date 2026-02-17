# source.py
# Универсальный источник видео: RTSP, файл, папка изображений

import cv2
import os
import time
import numpy as np
from typing import Optional, Tuple, List
from threading import Thread, Event
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum

from .decoder import HWDecoder, DecoderBackend


class SourceType(Enum):
    RTSP = "rtsp"
    VIDEO = "video"
    FOLDER = "folder"
    IMAGE = "image"


@dataclass
class SourceInfo:
    source_type: SourceType
    name: str
    width: int = 0
    height: int = 0
    fps: float = 30.0
    total_frames: int = 0
    backend: str = "None"


class VideoSource:
    """
    Универсальный источник видео с prefetch и auto-reconnect.

    Поддерживает:
    - RTSP потоки (с prefetch и reconnect)
    - Видеофайлы
    - Папки с изображениями
    - Одиночные изображения
    """

    def __init__(
        self,
        url: str,
        source_type: Optional[SourceType] = None,
        prefer_hw: bool = True,
        prefetch: bool = True,
        prefetch_size: int = 8,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
        no_drop: bool = False,
    ):
        self.url = url
        self.prefer_hw = prefer_hw
        self.prefetch_enabled = prefetch
        self.prefetch_size = prefetch_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.no_drop = no_drop

        # Auto-detect source type
        if source_type:
            self.source_type = source_type
        else:
            self.source_type = self._detect_type(url)

        self._decoder: Optional[HWDecoder] = None
        self._image_files: List[str] = []
        self._image_index = 0

        # Prefetch
        self._prefetch_queue: Queue = Queue(maxsize=prefetch_size)
        self._prefetch_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._reconnect_count = 0

        # Decode stats
        self._decode_fps = 0.0
        self._decode_count = 0
        self._decode_start = time.time()
        self._frames_dropped = 0

        # Info
        self.info: Optional[SourceInfo] = None

        # Open
        self._open()

    def _detect_type(self, url: str) -> SourceType:
        """Автоопределение типа источника"""
        if url.startswith("rtsp://"):
            return SourceType.RTSP
        elif os.path.isdir(url):
            return SourceType.FOLDER
        elif os.path.isfile(url):
            ext = os.path.splitext(url)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp"):
                return SourceType.IMAGE
            else:
                return SourceType.VIDEO
        else:
            # Assume video file path
            return SourceType.VIDEO

    def _open(self):
        """Открывает источник"""
        if self.source_type in (SourceType.RTSP, SourceType.VIDEO):
            self._open_video()
        elif self.source_type == SourceType.FOLDER:
            self._open_folder()
        elif self.source_type == SourceType.IMAGE:
            self._open_image()

    def _open_video(self):
        """Открывает видео/RTSP"""
        self._decoder = HWDecoder(prefer_hw=self.prefer_hw)
        decoder_info = self._decoder.open(self.url)

        if decoder_info.cap is None:
            raise RuntimeError(f"Failed to open: {self.url}")

        self.info = SourceInfo(
            source_type=self.source_type,
            name=self._get_name(),
            width=decoder_info.width,
            height=decoder_info.height,
            fps=decoder_info.fps,
            total_frames=int(decoder_info.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type == SourceType.VIDEO else 0,
            backend=decoder_info.backend.value,
        )

        # Prefetch для RTSP и видеофайлов
        if self.prefetch_enabled and self.source_type in (SourceType.RTSP, SourceType.VIDEO):
            self._start_prefetch()

    def _open_folder(self):
        """Открывает папку с изображениями"""
        self._image_files = sorted([
            os.path.join(self.url, f)
            for f in os.listdir(self.url)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        if not self._image_files:
            raise RuntimeError(f"No images in folder: {self.url}")

        # Читаем первое для размеров
        first = cv2.imread(self._image_files[0])
        h, w = first.shape[:2]

        self.info = SourceInfo(
            source_type=self.source_type,
            name=self._get_name(),
            width=w,
            height=h,
            fps=30.0,
            total_frames=len(self._image_files),
            backend="Images",
        )

    def _open_image(self):
        """Открывает одно изображение"""
        self._image_files = [self.url]

        img = cv2.imread(self.url)
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.url}")

        h, w = img.shape[:2]

        self.info = SourceInfo(
            source_type=self.source_type,
            name=self._get_name(),
            width=w,
            height=h,
            fps=1.0,
            total_frames=1,
            backend="Image",
        )

    def _get_name(self) -> str:
        """Имя источника"""
        if self.source_type == SourceType.RTSP:
            # Извлекаем IP или имя из URL
            return self.url.split("@")[-1].split("/")[0].split(":")[0]
        else:
            return os.path.basename(self.url).split(".")[0]

    # =========================================================
    # Prefetch (для RTSP)
    # =========================================================

    def _start_prefetch(self):
        """Запускает prefetch поток"""
        self._stop_event.clear()
        self._prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

    def _prefetch_loop(self):
        """Цикл prefetch с reconnect (RTSP) или EOF (видеофайл)"""
        consecutive_failures = 0
        max_consecutive_failures = 90  # переподключение после 90 неудачных чтений (~3 сек при 30fps)

        while not self._stop_event.is_set():
            if self._decoder is None or not self._decoder.is_opened():
                if self.source_type == SourceType.VIDEO:
                    self._prefetch_queue.put((False, None, 0.0))
                    break

                # RTSP: Reconnect
                if self._reconnect_count >= self.max_reconnects:
                    print(f"[VideoSource] Max reconnects reached")
                    break

                print(f"[VideoSource] Reconnecting ({self._reconnect_count + 1}/{self.max_reconnects})...")
                time.sleep(self.reconnect_delay)

                self._decoder = HWDecoder(prefer_hw=self.prefer_hw, verbose=False)
                decoder_info = self._decoder.open(self.url)

                if decoder_info.cap is None:
                    self._reconnect_count += 1
                    continue

                self._reconnect_count = 0
                consecutive_failures = 0
                print(f"[VideoSource] Reconnected via {decoder_info.backend.value}")

            # Read frame
            try:
                ok, frame = self._decoder.read()
            except Exception as e:
                print(f"[VideoSource] Read error: {e}")
                ok, frame = False, None

            capture_ts = time.time()

            if ok and frame is not None:
                consecutive_failures = 0

                # Decode FPS counter
                self._decode_count += 1
                now = time.time()
                if now - self._decode_start >= 1.0:
                    self._decode_fps = self._decode_count / (now - self._decode_start)
                    self._decode_count = 0
                    self._decode_start = now

                if self._prefetch_queue.full():
                    if self.source_type == SourceType.RTSP and not self.no_drop:
                        # RTSP realtime: drop oldest to keep live current
                        try:
                            self._prefetch_queue.get_nowait()
                            self._frames_dropped += 1
                        except Empty:
                            pass
                    else:
                        # Video file OR no_drop mode: wait for consumer (don't skip frames)
                        while self._prefetch_queue.full() and not self._stop_event.is_set():
                            time.sleep(0.001)

                self._prefetch_queue.put((True, frame, capture_ts))

            elif self.source_type == SourceType.VIDEO:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    # Real EOF or too many corrupted frames
                    self._prefetch_queue.put((False, None, 0.0))
                    break
            else:
                # RTSP: битый кадр или потеря соединения
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"[VideoSource] {consecutive_failures} consecutive read failures, reconnecting...")
                    self._decoder.close()
                    consecutive_failures = 0

    # =========================================================
    # Public API
    # =========================================================

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Читает кадр. Возвращает (ok, frame, capture_timestamp)"""
        # Prefetch mode
        if self._prefetch_thread is not None:
            try:
                return self._prefetch_queue.get(timeout=1.0)
            except Empty:
                return False, None, 0.0

        # Video file (no prefetch)
        if self._decoder is not None:
            ts = time.time()
            ok, frame = self._decoder.read()
            return ok, frame, ts

        # Images
        if self._image_index < len(self._image_files):
            ts = time.time()
            frame = cv2.imread(self._image_files[self._image_index])
            self._image_index += 1
            return True, frame, ts

        return False, None, 0.0

    @property
    def queue_size(self) -> int:
        """Текущий размер очереди prefetch"""
        return self._prefetch_queue.qsize()

    @property
    def queue_capacity(self) -> int:
        """Макс. размер очереди"""
        return self.prefetch_size

    @property
    def decode_fps(self) -> float:
        """FPS декодирования (в prefetch потоке)"""
        return self._decode_fps

    @property
    def frames_dropped(self) -> int:
        """Кол-во дропнутых кадров из очереди"""
        return self._frames_dropped

    def release(self):
        """Закрывает источник"""
        self._stop_event.set()

        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=2.0)
            self._prefetch_thread = None

        if self._decoder:
            self._decoder.close()
            self._decoder = None

    def is_opened(self) -> bool:
        if self._decoder:
            return self._decoder.is_opened()
        return self._image_index < len(self._image_files)

    def get_pos_frames(self) -> int:
        """Текущая позиция (для видео)"""
        if self._decoder and self._decoder._cap:
            return int(self._decoder._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return self._image_index

    def set_pos_frames(self, pos: int):
        """Установить позицию (для видео)"""
        if self._decoder and self._decoder._cap:
            self._decoder._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        elif self._image_files:
            self._image_index = min(pos, len(self._image_files) - 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# =========================================================
# Helper: создание из конфига камеры
# =========================================================

def make_rtsp_url(credentials: dict, camera: dict) -> str:
    """Создаёт RTSP URL из конфига"""
    cred = credentials
    ip = camera["ip"]
    stream_path = cred.get("stream_path", "")
    return f"rtsp://{cred['user']}:{cred['password']}@{ip}:{cred['port']}{stream_path}"


def create_source_from_config(
    cfg_cam: dict,
    camera_name: str,
    prefer_hw: bool = True,
    no_drop: bool = False,
    prefetch_size: int = 8,
) -> VideoSource:
    """Создаёт VideoSource из config_cam.yaml"""
    cred = cfg_cam.get("camera_credentials", {})
    cameras = cfg_cam.get("cameras", [])

    cam = next((c for c in cameras if c["name"] == camera_name), None)
    if not cam:
        raise ValueError(f"Camera {camera_name} not found in config")

    url = make_rtsp_url(cred, cam)
    source = VideoSource(
        url, source_type=SourceType.RTSP, prefer_hw=prefer_hw,
        no_drop=no_drop, prefetch_size=prefetch_size,
    )
    source.info.name = camera_name  # Use config name (e.g. "Camera_21"), not IP
    return source
