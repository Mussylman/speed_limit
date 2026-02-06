# decoder.py
# Аппаратное декодирование: NVDEC → GStreamer → FFmpeg fallback

import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DecoderBackend(Enum):
    NVDEC = "NVDEC"
    GSTREAMER = "GStreamer"
    FFMPEG = "FFmpeg"
    NONE = "None"


@dataclass
class DecoderInfo:
    backend: DecoderBackend
    cap: Optional[cv2.VideoCapture]
    width: int = 0
    height: int = 0
    fps: float = 0.0


class HWDecoder:
    """
    Аппаратный декодер видео с автоматическим fallback:
    NVDEC (CUDA) → GStreamer → FFmpeg (CPU)
    """

    def __init__(self, prefer_hw: bool = True, verbose: bool = True):
        self.prefer_hw = prefer_hw
        self.verbose = verbose
        self.current_backend = DecoderBackend.NONE
        self._cap: Optional[cv2.VideoCapture] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Decoder] {msg}")

    def _try_nvdec(self, url: str) -> Optional[cv2.VideoCapture]:
        """NVDEC через OpenCV CUDA"""
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._log("NVDEC: OK")
                    return cap
            cap.release()
        except Exception as e:
            self._log(f"NVDEC failed: {e}")
        return None

    def _try_gstreamer_hw(self, url: str) -> Optional[cv2.VideoCapture]:
        """GStreamer с nvh264dec"""
        try:
            if url.startswith("rtsp://"):
                pipeline = (
                    f'rtspsrc location="{url}" latency=100 ! '
                    f'rtph264depay ! h264parse ! nvh264dec ! '
                    f'videoconvert ! video/x-raw,format=BGR ! '
                    f'appsink drop=1 max-buffers=2'
                )
            else:
                pipeline = (
                    f'filesrc location="{url}" ! '
                    f'decodebin ! nvh264dec ! '
                    f'videoconvert ! video/x-raw,format=BGR ! '
                    f'appsink drop=1 max-buffers=2'
                )

            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._log("GStreamer+NVDEC: OK")
                    return cap
            cap.release()
        except Exception as e:
            self._log(f"GStreamer HW failed: {e}")
        return None

    def _try_gstreamer_sw(self, url: str) -> Optional[cv2.VideoCapture]:
        """GStreamer software decoding"""
        try:
            if url.startswith("rtsp://"):
                pipeline = (
                    f'rtspsrc location="{url}" latency=100 ! '
                    f'decodebin ! videoconvert ! '
                    f'video/x-raw,format=BGR ! appsink drop=1'
                )
            else:
                pipeline = (
                    f'filesrc location="{url}" ! '
                    f'decodebin ! videoconvert ! '
                    f'video/x-raw,format=BGR ! appsink drop=1'
                )

            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._log("GStreamer (CPU): OK")
                    return cap
            cap.release()
        except Exception as e:
            self._log(f"GStreamer SW failed: {e}")
        return None

    def _try_ffmpeg(self, url: str) -> Optional[cv2.VideoCapture]:
        """FFmpeg fallback"""
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._log("FFmpeg: OK")
                    return cap
            cap.release()
        except Exception as e:
            self._log(f"FFmpeg failed: {e}")
        return None

    def open(self, url: str) -> DecoderInfo:
        """Открывает видео с автоматическим выбором декодера"""
        self.close()

        backends = [
            (DecoderBackend.NVDEC, self._try_nvdec),
            (DecoderBackend.GSTREAMER, self._try_gstreamer_hw),
            (DecoderBackend.GSTREAMER, self._try_gstreamer_sw),
            (DecoderBackend.FFMPEG, self._try_ffmpeg),
        ]

        if not self.prefer_hw:
            backends = [(DecoderBackend.FFMPEG, self._try_ffmpeg)]

        for backend, try_fn in backends:
            cap = try_fn(url)
            if cap is not None:
                self._cap = cap
                self.current_backend = backend

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                return DecoderInfo(
                    backend=backend,
                    cap=cap,
                    width=width,
                    height=height,
                    fps=fps,
                )

        self._log(f"All backends failed for: {url}")
        self.current_backend = DecoderBackend.NONE
        return DecoderInfo(backend=DecoderBackend.NONE, cap=None)

    def read(self) -> Tuple[bool, Optional[any]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.current_backend = DecoderBackend.NONE

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_backend_name(self) -> str:
        return self.current_backend.value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
