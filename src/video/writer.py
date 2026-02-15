# video/writer.py
# Async video writing via FFmpeg subprocess or OpenCV fallback

import cv2
import numpy as np
from typing import Tuple
from threading import Thread
from queue import Queue


class AsyncVideoWriter:
    """Async video writing via FFmpeg subprocess.

    Priority: h264_nvenc (GPU) -> libx264 (CPU) -> mp4v (OpenCV).
    Produces WhatsApp/Telegram compatible H.264 MP4.
    """

    def __init__(self, path: str, fps: float, size: Tuple[int, int], bitrate_kbps: int = 8000):
        import subprocess, shutil

        self.w, self.h = size
        self.path = path
        self.proc = None
        self.encoder_name = "none"

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            self.proc = None
            self._init_opencv_fallback(path, fps, size)
            return

        # Try NVENC first, then libx264
        for enc, name in [("h264_nvenc", "NVENC (GPU)"), ("libx264", "libx264 (CPU)")]:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.w}x{self.h}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", enc,
                "-b:v", f"{bitrate_kbps}k",
                "-maxrate", f"{bitrate_kbps * 2}k",
                "-bufsize", f"{bitrate_kbps * 2}k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                path,
            ]
            try:
                self.proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                import time; time.sleep(0.3)
                if self.proc.poll() is not None:
                    self.proc = None
                    continue
                self.encoder_name = name
                break
            except Exception:
                self.proc = None

        if self.proc is None:
            self._init_opencv_fallback(path, fps, size)
            return

        print(f"  VideoWriter: {self.encoder_name} ({bitrate_kbps} kbps)")

        self.queue: Queue = Queue(maxsize=120)
        self.running = True
        self.thread = Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _init_opencv_fallback(self, path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._cv_writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.encoder_name = "mp4v (OpenCV fallback)"
        print(f"  VideoWriter: {self.encoder_name}")
        self.queue: Queue = Queue(maxsize=120)
        self.running = True
        self.thread = Thread(target=self._write_loop_cv, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                if self.proc and self.proc.stdin:
                    self.proc.stdin.write(frame.tobytes())
            except Exception:
                pass

    def _write_loop_cv(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self._cv_writer.write(frame)
            except Exception:
                pass

    def write(self, frame: np.ndarray):
        if not self.queue.full():
            self.queue.put(frame.copy())

    def release(self):
        self.running = False
        self.thread.join(timeout=10.0)
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        elif hasattr(self, '_cv_writer'):
            self._cv_writer.release()
