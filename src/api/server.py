# server.py
# FastAPI сервер

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    CameraInfo, CameraLocation, CameraStartRequest, CameraStatus,
    ViolationList, VehicleList, HealthResponse,
)
from .websocket import ws_manager
from .storage import OutputStorage
from .mjpeg import mjpeg_streamer
from .integration import pipeline_events

# Пути
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
CONFIG_CAM_PATH = BASE_DIR / "config" / "config_cam.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Storage
storage = OutputStorage(str(OUTPUTS_DIR))


# ============================================================
# Helpers
# ============================================================

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_cameras_config() -> dict:
    if CONFIG_CAM_PATH.exists():
        with open(CONFIG_CAM_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def make_rtsp_url(cred: dict, cam: dict) -> str:
    stream_path = cred.get("stream_path", "")
    return f"rtsp://{cred['user']}:{cred['password']}@{cam['ip']}:{cred['port']}{stream_path}"


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Starting...")
    pipeline_events.start()
    yield
    print("[API] Shutting down...")
    mjpeg_streamer.stop_all()
    pipeline_events.stop()


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Speed Limit API",
    description="API для системы измерения скорости и распознавания номеров",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# ============================================================
# Cameras
# ============================================================

@app.get("/api/cameras", response_model=List[CameraInfo])
async def get_cameras():
    """Список камер"""
    cfg = load_cameras_config()
    cred = cfg.get("camera_credentials", {})
    cameras = cfg.get("cameras", [])

    # Координаты камер в Шымкенте (примерные позиции вдоль основных дорог)
    default_locations = {
        "Camera_100": {"lat": 42.3200, "lng": 69.5950, "address": "ул. Тауке хана"},
        "Camera_101": {"lat": 42.3250, "lng": 69.5850, "address": "пр. Республики"},
        "Camera_102": {"lat": 42.3300, "lng": 69.5750, "address": "ул. Байтурсынова"},
        "Camera_103": {"lat": 42.3350, "lng": 69.5900, "address": "ул. Кунаева"},
        "Camera_104": {"lat": 42.3400, "lng": 69.5800, "address": "пр. Тауелсиздик"},
        "Camera_105": {"lat": 42.3450, "lng": 69.5700, "address": "ул. Жибек жолы"},
        "Camera_106": {"lat": 42.3380, "lng": 69.6000, "address": "ул. Туркестанская"},
        "Camera_107": {"lat": 42.3280, "lng": 69.6050, "address": "ул. Иляева"},
        "Camera_108": {"lat": 42.3150, "lng": 69.5850, "address": "ул. Мадели кожа"},
        "Camera_109": {"lat": 42.3500, "lng": 69.5950, "address": "ул. Казыбек би"},
        "Camera_110": {"lat": 42.3220, "lng": 69.6100, "address": "пр. Кунаева"},
    }

    result = []
    for cam in cameras:
        cam_id = cam.get("name", "unknown")
        cam_ip = cam.get("ip", "")
        status = mjpeg_streamer.get_status(cam_id)
        is_running = status and status.get("running", False)

        # Build RTSP URL
        rtsp_url = None
        if cred and cam_ip:
            rtsp_url = make_rtsp_url(cred, cam)

        # Location
        loc_data = default_locations.get(cam_id, {"lat": 42.3417, "lng": 69.5901})
        location = CameraLocation(
            lat=loc_data.get("lat", 42.3417),
            lng=loc_data.get("lng", 69.5901),
            address=loc_data.get("address"),
        )

        result.append(CameraInfo(
            id=cam_id,
            name=cam.get("name", cam_id),
            ip=cam_ip,
            rtsp_url=rtsp_url,
            hls_url=f"/api/stream/{cam_id}/mjpeg" if is_running else None,
            location=location,
            type="smart",
            status="online" if is_running else "offline",
            backend=status["backend"] if status else None,
            fps=status["fps"] if status else 0,
        ))

    return result


@app.get("/api/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """Статус камеры"""
    status = mjpeg_streamer.get_status(camera_id)
    if not status:
        raise HTTPException(404, f"Camera {camera_id} not running")
    return status


@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str, request: CameraStartRequest):
    """Запуск стрима"""
    url = request.url

    if not url:
        cfg_cam = load_cameras_config()
        cred = cfg_cam.get("camera_credentials", {})
        cameras = cfg_cam.get("cameras", [])

        cam = next((c for c in cameras if c["name"] == camera_id), None)
        if not cam:
            raise HTTPException(404, f"Camera {camera_id} not found")

        url = make_rtsp_url(cred, cam)

    success = mjpeg_streamer.start(camera_id, url)

    if success:
        await ws_manager.broadcast("system", "camera_started", {"camera_id": camera_id})
        return {"status": "started", "camera_id": camera_id}

    raise HTTPException(500, f"Failed to start {camera_id}")


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Остановка стрима"""
    success = mjpeg_streamer.stop(camera_id)

    if success:
        await ws_manager.broadcast("system", "camera_stopped", {"camera_id": camera_id})
        return {"status": "stopped", "camera_id": camera_id}

    raise HTTPException(404, f"Camera {camera_id} not running")


# ============================================================
# Demo — тестовое видео
# ============================================================

@app.post("/api/demo/start")
async def start_demo():
    """Запуск демо-режима с тестовым видео + pipeline (YOLO + скорость + OCR)"""
    from .pipeline_processor import get_or_create_pipeline

    video_dir = BASE_DIR / "videos"
    video_files = list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4"))

    if not video_files:
        raise HTTPException(404, "No test videos found in videos/")

    video_path = str(video_files[0])

    cfg = load_config()
    cfg_cam = load_cameras_config()
    cameras = cfg_cam.get("cameras", [])

    # Match camera to video filename (e.g. Camera_108.avi -> Camera_108)
    video_name = Path(video_path).stem  # "Camera_108"
    target_cam = next((c for c in cameras if c.get("name") == video_name), None)
    if not target_cam and cameras:
        target_cam = cameras[0]

    started = []
    if target_cam:
        cam_id = target_cam.get("name", "unknown")
        # Get actual video FPS for correct speed calculation
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(video_path)
        source_fps = _cap.get(_cv2.CAP_PROP_FPS) if _cap.isOpened() else 0
        _cap.release()
        pipeline = get_or_create_pipeline(cfg, cfg_cam, cam_id, source_fps=source_fps)
        success = mjpeg_streamer.start(cam_id, video_path, frame_processor=pipeline.process_frame)
        if success:
            started.append(cam_id)
            await ws_manager.broadcast("system", "camera_started", {"camera_id": cam_id})

    return {"status": "demo_started", "cameras": started, "video": video_path}


@app.post("/api/demo/stop")
async def stop_demo():
    """Остановка демо"""
    from .pipeline_processor import stop_pipeline

    streams = mjpeg_streamer.list_streams()
    for cam_id in streams:
        mjpeg_streamer.stop(cam_id)
        stop_pipeline(cam_id)
    return {"status": "demo_stopped", "cameras": streams}


# ============================================================
# Violations
# ============================================================

@app.get("/api/violations")
async def get_violations(
    camera_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Нарушения"""
    items, total = storage.get_violations(camera_id, limit, offset)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
    }


# ============================================================
# Vehicles
# ============================================================

@app.get("/api/vehicles")
async def get_vehicles(
    camera_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Распознанные номера"""
    items, total = storage.get_vehicles(camera_id, limit, offset)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
    }


# ============================================================
# Speeds
# ============================================================

@app.get("/api/speeds")
async def get_speeds(
    camera_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Измерения скорости"""
    return storage.get_speeds(camera_id, limit)


# ============================================================
# Streaming
# ============================================================

@app.get("/api/stream/{camera_id}/mjpeg")
async def stream_mjpeg(camera_id: str):
    """MJPEG поток"""
    if camera_id not in mjpeg_streamer.list_streams():
        raise HTTPException(404, f"Stream {camera_id} not running")

    return StreamingResponse(
        mjpeg_streamer.generate(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/{camera_id}/snapshot")
async def stream_snapshot(camera_id: str):
    """Текущий кадр"""
    jpeg = mjpeg_streamer.get_snapshot(camera_id)

    if jpeg is None:
        raise HTTPException(404, f"No frame for {camera_id}")

    return StreamingResponse(
        iter([jpeg]),
        media_type="image/jpeg",
    )


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time"""
    await ws_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


@app.get("/api/ws/stats")
async def ws_stats():
    """Статистика WS"""
    return ws_manager.get_stats()


# ============================================================
# System
# ============================================================

@app.get("/api/config")
async def get_config():
    """Конфиг (без секретов)"""
    cfg = load_config()
    cfg.pop("camera_credentials", None)
    return cfg


@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "streams": mjpeg_streamer.list_streams(),
        "ws_connections": ws_manager.get_stats()["total_connections"],
    }


@app.get("/api/metrics/{camera_id}")
async def get_metrics(camera_id: str, n: int = Query(100, ge=1, le=1000)):
    """Последние N метрик"""
    return storage.tail_jsonl(camera_id, "metrics.jsonl", n)


@app.get("/api/summary/{camera_id}")
async def get_summary(camera_id: str):
    """Summary метрик"""
    summary = storage.get_summary(camera_id)
    if not summary:
        raise HTTPException(404, f"No summary for {camera_id}")
    return summary


# ============================================================
# Run
# ============================================================

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
