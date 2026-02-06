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
        prefetch_size: int = 3,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
    ):
        self.url = url
        self.prefer_hw = prefer_hw
        self.prefetch_enabled = prefetch
        self.prefetch_size = prefetch_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects

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

        # Prefetch только для RTSP
        if self.prefetch_enabled and self.source_type == SourceType.RTSP:
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
        """Цикл prefetch с reconnect"""
        while not self._stop_event.is_set():
            if self._decoder is None or not self._decoder.is_opened():
                # Reconnect
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
                print(f"[VideoSource] Reconnected via {decoder_info.backend.value}")

            # Read frame
            if not self._prefetch_queue.full():
                ok, frame = self._decoder.read()
                if ok and frame is not None:
                    self._prefetch_queue.put((True, frame))
                else:
                    # Connection lost
                    self._decoder.close()
            else:
                time.sleep(0.001)

    # =========================================================
    # Public API
    # =========================================================

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Читает кадр"""
        # Prefetch mode (RTSP)
        if self._prefetch_thread is not None:
            try:
                return self._prefetch_queue.get(timeout=1.0)
            except Empty:
                return False, None

        # Video file
        if self._decoder is not None:
            return self._decoder.read()

        # Images
        if self._image_index < len(self._image_files):
            frame = cv2.imread(self._image_files[self._image_index])
            self._image_index += 1
            return True, frame

        return False, None

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
) -> VideoSource:
    """Создаёт VideoSource из config_cam.yaml"""
    cred = cfg_cam.get("camera_credentials", {})
    cameras = cfg_cam.get("cameras", [])

    cam = next((c for c in cameras if c["name"] == camera_name), None)
    if not cam:
        raise ValueError(f"Camera {camera_name} not found in config")

    url = make_rtsp_url(cred, cam)
    return VideoSource(url, source_type=SourceType.RTSP, prefer_hw=prefer_hw)
