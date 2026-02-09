# schemas.py
# Pydantic схемы для API

from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================
# Камеры
# ============================================================

class CameraLocation(BaseModel):
    lat: float = 0.0
    lng: float = 0.0
    address: Optional[str] = None


class CameraInfo(BaseModel):
    id: str
    name: str
    ip: Optional[str] = None
    rtsp_url: Optional[str] = None
    hls_url: Optional[str] = None
    location: CameraLocation = CameraLocation()
    type: str = "standard"  # smart, standard
    status: str = "offline"  # online, offline, error
    backend: Optional[str] = None  # NVDEC, GStreamer, FFmpeg
    fps: float = 0
    width: int = 0
    height: int = 0
    lane: Optional[int] = None


class CameraStartRequest(BaseModel):
    url: Optional[str] = None


class CameraStatus(BaseModel):
    camera_id: str
    running: bool
    backend: str
    fps: float
    frames_read: int
    frames_served: int
    reconnects: int
    errors: int
    started_at: Optional[str] = None


# ============================================================
# Нарушения
# ============================================================

class ViolationRecord(BaseModel):
    id: Optional[str] = None
    camera_id: str
    track_id: int
    plate: str
    speed_kmh: float
    limit_kmh: float
    over_speed: float = 0
    timestamp: str
    image_path: Optional[str] = None


class ViolationList(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[ViolationRecord]


# ============================================================
# Транспорт / Номера
# ============================================================

class VehicleRecord(BaseModel):
    track_id: int
    plate: str
    confidence: float
    speed_kmh: Optional[float] = None
    camera_id: str
    timestamp: str
    image_path: Optional[str] = None


class VehicleList(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[VehicleRecord]


# ============================================================
# Скорости
# ============================================================

class SpeedRecord(BaseModel):
    track_id: int
    speed_kmh: float
    camera_id: str
    timestamp: str
    plate: Optional[str] = None


# ============================================================
# WebSocket события
# ============================================================

class WSSpeedUpdate(BaseModel):
    track_id: int
    speed_kmh: float
    camera_id: str
    timestamp: str


class WSViolation(BaseModel):
    track_id: int
    plate: str
    speed_kmh: float
    limit_kmh: float
    over_speed: float
    camera_id: str
    image_path: Optional[str] = None
    timestamp: str


class WSPlateRecognized(BaseModel):
    track_id: int
    plate: str
    confidence: float
    speed_kmh: Optional[float] = None
    camera_id: str
    timestamp: str


class WSStats(BaseModel):
    frame_idx: int
    detections: int
    fps: float
    camera_id: str
    timestamp: str


# ============================================================
# Система
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str
    streams: List[str]
    ws_connections: int


class ConfigResponse(BaseModel):
    device: str
    fps: int
    frame_skip: int
    speed_method: str
    speed_limit: float
    yolo_imgsz: int
