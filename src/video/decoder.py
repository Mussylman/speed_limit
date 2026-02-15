# decoder.py
# Аппаратное декодирование: NVDEC → GStreamer → FFmpeg fallback
# Буфер RTSP настраивается автоматически по пингу

import os
import re
import subprocess
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


def _ping_ms(host: str, timeout_s: int = 3) -> Optional[float]:
    """Пинг хоста, возвращает RTT в мс или None если недоступен."""
    try:
        # Windows: ping -n 4 -w <timeout_ms>
        result = subprocess.run(
            ["ping", "-n", "4", "-w", str(timeout_s * 1000), host],
            capture_output=True, text=True, timeout=timeout_s + 6,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        # Ищем "Average = Xms" (en) или "Среднее = Xмсек" (ru)
        # м = кириллическая, m = латинская
        m = re.search(r'[=<]\s*(\d+)\s*[мm]', result.stdout, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _apply_rtsp_buffer(host: str):
    """Устанавливает OPENCV_FFMPEG_CAPTURE_OPTIONS по пингу к камере.

    Пинг < 10мс  (LAN)        → буфер 2MB,  max_delay 200мс
    Пинг 10-50мс (локальная)   → буфер 8MB,  max_delay 500мс
    Пинг 50-200мс (интернет)   → буфер 32MB, max_delay 2000мс
    Пинг > 200мс (плохая сеть) → буфер 64MB, max_delay 5000мс
    Недоступен                  → максимальный буфер
    """
    rtt = _ping_ms(host)

    if rtt is not None and rtt < 10:
        # LAN — камера рядом
        buf, delay, probe = 2 * 1024 * 1024, 200_000, 1_000_000
        label = f"LAN ({rtt:.0f}ms)"
    elif rtt is not None and rtt < 50:
        # Ближний интернет / VPN
        buf, delay, probe = 8 * 1024 * 1024, 500_000, 2_000_000
        label = f"near ({rtt:.0f}ms)"
    elif rtt is not None and rtt < 200:
        # Удалённый интернет
        buf, delay, probe = 32 * 1024 * 1024, 2_000_000, 5_000_000
        label = f"remote ({rtt:.0f}ms)"
    else:
        # Плохая сеть (>200ms) или недоступен
        buf, delay, probe = 64 * 1024 * 1024, 5_000_000, 10_000_000
        label = f"bad link ({rtt:.0f}ms)" if rtt else "unreachable"

    opts = f"rtsp_transport;tcp|buffer_size;{buf}|max_delay;{delay}|analyzeduration;{probe}|probesize;{probe}"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
    print(f"[Decoder] Ping {host}: {label} → buffer={buf // (1024*1024)}MB, delay={delay // 1000}ms")


# Дефолт для файлов (не RTSP)
if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2097152|max_delay;200000"


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
                    f'rtspsrc location="{url}" latency=2000 protocols=tcp ! '
                    f'rtph264depay ! h264parse ! nvh264dec ! '
                    f'videoconvert ! video/x-raw,format=BGR ! '
                    f'appsink drop=0 max-buffers=64 sync=false'
                )
            else:
                pipeline = (
                    f'filesrc location="{url}" ! '
                    f'decodebin ! nvh264dec ! '
                    f'videoconvert ! video/x-raw,format=BGR ! '
                    f'appsink drop=0 max-buffers=64 sync=false'
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
                    f'rtspsrc location="{url}" latency=2000 protocols=tcp ! '
                    f'decodebin ! videoconvert ! '
                    f'video/x-raw,format=BGR ! appsink drop=0 max-buffers=64 sync=false'
                )
            else:
                pipeline = (
                    f'filesrc location="{url}" ! '
                    f'decodebin ! videoconvert ! '
                    f'video/x-raw,format=BGR ! appsink drop=0 max-buffers=64 sync=false'
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

        # Авто-буфер по пингу для RTSP
        if url.startswith("rtsp://"):
            # Извлекаем IP из rtsp://user:pass@IP:port/path
            try:
                host = url.split("@")[-1].split(":")[0].split("/")[0]
                _apply_rtsp_buffer(host)
            except Exception:
                pass

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
