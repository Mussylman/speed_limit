# main.py
# Pipeline: YOLO трекинг + скорость + OCR номеров (NomeroffNet)
# OCR каждый кадр, сохраняется лучший по confidence

import os
import sys
import cv2
import yaml
import time
import argparse
import datetime
import numpy as np
from typing import Dict, Tuple
from threading import Thread
from queue import Queue
from ultralytics import YOLO


class AsyncVideoWriter:
    """Асинхронная запись видео в отдельном потоке"""
    def __init__(self, path: str, fourcc: int, fps: float, size: Tuple[int, int]):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.queue: Queue = Queue(maxsize=30)  # буфер на 30 кадров
        self.running = True
        self.thread = Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
            except:
                pass

    def write(self, frame: np.ndarray):
        if not self.queue.full():
            self.queue.put(frame.copy())  # copy потому что frame может измениться

    def release(self):
        self.running = False
        self.thread.join(timeout=5.0)
        self.writer.release()

# Headless режим (Docker без дисплея)
HEADLESS = os.environ.get("HEADLESS", "0") == "1" or os.environ.get("DISPLAY_OFF", "0") == "1"

# Базовая директория проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "nomeroff-net"))

from speed_line import SpeedEstimator
from speed_homography import HomographySpeedEstimator
from plate_recognizer import PlateRecognizer
from video.source import VideoSource, SourceType, create_source_from_config
from speed_logger import SpeedLogger
from metrics_logger import MetricsLogger
from async_ocr import AsyncOCR
from async_yolo import AsyncYOLO
from file_logger import FileLogger

import torch


def print_gpu_info():
    """Выводит информацию о GPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  CUDA недоступна, используется CPU")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_results_panel(frame, passed_results, speed_logger, speed_limit):
    """Рисует панель с номерами и скоростями слева"""
    h, w = frame.shape[:2]
    panel_width = 280

    # Быстрый тёмный фон без alpha-blend
    cv2.rectangle(frame, (0, 0), (panel_width, h), (30, 30, 30), -1)

    # Заголовок
    cv2.putText(frame, "PASSED VEHICLES", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(frame, (10, 50), (panel_width - 10, 50), (100, 100, 100), 1)

    # Список последних 10 результатов
    y_offset = 80
    items = list(passed_results.items())[-10:]  # последние 10

    for track_id, event in reversed(items):
        plate = event.plate_text

        # Получаем скорость из speed_logger
        speed_event = speed_logger.speeds.get(track_id)
        spd = speed_event.speed_kmh if speed_event else 0

        # Цвет скорости
        if spd > speed_limit:
            speed_color = (0, 0, 255)  # красный - нарушение
            status = "!"
        else:
            speed_color = (0, 255, 0)  # зелёный - норма
            status = ""

        # Номер
        cv2.putText(frame, f"{plate}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Скорость
        speed_text = f"{spd:.0f} km/h {status}"
        cv2.putText(frame, speed_text, (15, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)

        y_offset += 60

        if y_offset > h - 50:
            break

    # Счётчик внизу
    total = len(passed_results)
    violations = sum(1 for tid in passed_results
                     if tid in speed_logger.speeds
                     and speed_logger.speeds[tid].speed_kmh > speed_limit)

    cv2.putText(frame, f"Total: {total}  Violations: {violations}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def load_zones_for_source(source, cfg, cam_cfg, speed):
    """Загружает зоны для измерения скорости."""
    line_zones = []

    if source.source_type == SourceType.RTSP and source.info:
        for cam in cam_cfg.get("cameras", []):
            if cam.get("name") == source.info.name:
                line_zones = cam.get("line_zones", [])
                break
    elif "line_zones" in cfg:
        line_zones = cfg["line_zones"]

    if line_zones:
        for zone in line_zones:
            speed.add_line_zone(
                zone["start_line"],
                zone["end_line"],
                zone.get("distance_m", 10),
                direction=zone.get("direction", "down"),
                name=zone.get("name", "Zone"),
                color=tuple(zone.get("color", [0, 255, 0])),
            )
        print(f"📏 Загружены зоны: {[z.get('name', 'Zone') for z in line_zones]}")

    return line_zones


def process_source(source, cfg, cam_cfg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = source.info.name
    out_dir = os.path.join(BASE_DIR, cfg["output_dir"], f"{name}_run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🚦 Старт: {name}")
    print(f"💾 Выход: {out_dir}")
    print(f"🎥 Backend: {source.info.backend}")

    # YOLO
    yolo_path = os.path.join(BASE_DIR, cfg["models"]["yolo_model"])
    model = YOLO(yolo_path, task="detect")
    try:
        model.to(cfg["device"])
    except:
        pass

    # FP16 для ускорения (если GPU поддерживает)
    use_half = cfg["device"] == "cuda" and torch.cuda.is_available()
    if use_half:
        print("⚡ YOLO: FP16 mode enabled")

    # FPS
    fps = source.info.fps if source.info.fps > 0 else cfg.get("fps", 30)

    # Скорость
    speed_method = cfg.get("speed_method", "lines")
    show_bird_eye = False

    if speed_method == "homography":
        hom_cfg = cfg.get("homography", {})
        hom_file = os.path.join(BASE_DIR, hom_cfg.get("config_file", "config/homography_config.yaml"))

        if os.path.exists(hom_file):
            speed = HomographySpeedEstimator(
                homography_config=hom_file,
                fps=fps,
                min_track_points=hom_cfg.get("min_track_points", 5),
                smoothing_window=hom_cfg.get("smoothing_window", 10),
                max_speed_kmh=hom_cfg.get("max_speed_kmh", 200),
                min_speed_kmh=hom_cfg.get("min_speed_kmh", 5),
                speed_correction=hom_cfg.get("speed_correction", 1.0),
            )
            show_bird_eye = hom_cfg.get("show_bird_eye", True)
            print(f"🎯 Скорость: ГОМОГРАФИЯ")
        else:
            speed_method = "lines"

    if speed_method == "lines":
        speed = SpeedEstimator(fps)
        load_zones_for_source(source, cfg, cam_cfg, speed)
        print(f"📏 Скорость: ЛИНИИ")

    # OCR (NomeroffNet) с фильтрами
    ocr_conf_threshold = float(cfg.get("ocr_conf_threshold", 0.5))
    plate_format = cfg.get("plate_format_regex", "")
    recognizer = PlateRecognizer(
        output_dir=out_dir,
        camera_id=name,
        min_conf=ocr_conf_threshold,
        min_plate_chars=int(cfg.get("min_plate_chars", 8)),
        min_car_height=int(cfg.get("min_car_height", 150)),
        min_car_width=int(cfg.get("min_car_width", 100)),
        min_plate_width=int(cfg.get("min_plate_width", 60)),
        min_plate_height=int(cfg.get("min_plate_height", 15)),
        cooldown_frames=int(cfg.get("ocr_cooldown_frames", 3)),
        plate_format_regex=plate_format,
    )

    # Логгер скорости
    speed_limit = int(cfg.get("speed_limit", 70))
    speed_logger = SpeedLogger(
        output_dir=out_dir,
        camera_id=name,
        speed_limit=speed_limit,
    )

    # Асинхронный OCR с оптимизациями
    async_ocr = AsyncOCR(
        recognizer,
        max_queue_size=64,           # [1] очередь с лимитом
        num_workers=1,
        max_crop_width=320,          # [4] resize перед очередью
        good_conf_threshold=0.88,    # [3] early stop если хороший
        cache_max_size=200,          # [5] лимит кэша
        cache_ttl_frames=300,        # [5] TTL ~10 сек при 30fps
    )

    print(f"🔤 OCR: NomeroffNet (ASYNC)")
    print(f"   Машина: >= {cfg.get('min_car_height', 150)}x{cfg.get('min_car_width', 100)} px")
    print(f"   Номер:  >= {cfg.get('min_plate_width', 60)}x{cfg.get('min_plate_height', 15)} px")
    print(f"   Формат: {plate_format if plate_format else 'любой'}")
    print(f"   Cooldown: {cfg.get('ocr_cooldown_frames', 3)} кадров")

    # Параметры
    yolo_imgsz = int(cfg.get("yolo_imgsz", 1280))
    frame_skip = int(cfg.get("frame_skip", 1))
    show_window = bool(cfg.get("show_window", True)) and not HEADLESS

    print(f"🔍 YOLO: imgsz={yolo_imgsz}, skip={frame_skip}")

    # Метрики (без live-line, всё в файлы)
    metrics = MetricsLogger(
        output_dir=out_dir,
        report_interval=10.0,  # отчёт каждые 10 сек
        json_log=True,
    )
    metrics.async_ocr = async_ocr

    # Файловый логгер (детальные логи)
    file_logger = FileLogger(output_dir=out_dir)
    recognizer.file_logger = file_logger  # для логирования OCR

    # Асинхронный YOLO
    async_yolo = AsyncYOLO(
        model=model,
        imgsz=yolo_imgsz,
        classes=[2],
        half=use_half,
        tracker="bytetrack.yaml",
    )
    print(f"⚡ YOLO: async mode (frame_skip={frame_skip})")

    print("🚗 Запуск... (Q — выход)\n")

    # Запись видео
    records_dir = os.path.join(BASE_DIR, "records")
    os.makedirs(records_dir, exist_ok=True)
    video_out_path = os.path.join(records_dir, f"{name}_{timestamp}.mp4")
    video_writer = None

    frame_idx = 0

    # Кэш последних детекций для плавного видео
    cached_detections = []  # [{box, obj_id, plate_text, plate_conf, speed}, ...]
    last_yolo_frame = 0
    last_yolo_time_ms = 0

    # Контроль FPS показа
    target_frame_time = 1.0 / fps
    last_display_time = time.time()

    while True:
        frame_start_time = time.time()

        ok, frame = source.read()

        if not ok or frame is None:
            if show_window:
                cv2.imshow(name, np.zeros((360, 640, 3), dtype=np.uint8))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        frame_idx += 1

        # === ОТПРАВКА НА YOLO (асинхронно, каждый N-й кадр) ===
        if frame_skip <= 1 or frame_idx % frame_skip == 0:
            async_yolo.submit(frame, frame_idx)

        # === ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ YOLO (неблокирующий) ===
        for yolo_result in async_yolo.get_results():
            metrics.start_frame(yolo_result.frame_idx)
            last_yolo_time_ms = yolo_result.processing_time_ms
            last_yolo_frame = yolo_result.frame_idx

            # Очищаем кэш детекций
            cached_detections = []

            # Детекции уже отфильтрованы и с crops
            detections = yolo_result.detections
            confs_list = [d.conf for d in detections]
            metrics.log_yolo(yolo_result.processing_time_ms, len(detections), confs_list)

            # Логируем детекции в файл
            file_logger.log_detections(
                frame_idx=yolo_result.frame_idx,
                detections=[{"obj_id": d.obj_id, "box": d.box, "conf": d.conf} for d in detections],
                yolo_time_ms=yolo_result.processing_time_ms,
            )

            for det in detections:
                obj_id = det.obj_id
                x1i, y1i, x2i, y2i = det.box

                # Скорость
                speed.process(yolo_result.frame_idx, obj_id, det.cx, det.cy)

                # OCR (асинхронный) - crop уже вырезан в YOLO thread
                async_ocr.submit(
                    track_id=obj_id,
                    crop=det.crop,
                    car_height=y2i - y1i,
                    car_width=x2i - x1i,
                    frame_idx=yolo_result.frame_idx,
                    detection_conf=det.conf,
                )

                # Получаем OCR из кэша
                cached_result = async_ocr.get_cached_result(obj_id)
                plate_text = cached_result.plate_text if cached_result else ""
                plate_conf = cached_result.ocr_conf if cached_result else 0.0

                # Скорость
                current_speed = None
                if speed_method == "homography":
                    v = speed.get_speed(obj_id)
                    if v:
                        current_speed = v
                else:
                    if hasattr(speed, 'object_speeds') and obj_id in speed.object_speeds:
                        v, _ = speed.object_speeds[obj_id]
                        current_speed = v

                # Логируем скорость
                if current_speed:
                    is_violation = current_speed > speed_limit
                    speed_logger.update(
                        track_id=obj_id,
                        speed_kmh=current_speed,
                        frame_idx=yolo_result.frame_idx,
                        plate_text=plate_text,
                        plate_conf=plate_conf,
                    )
                    metrics.log_speed(current_speed, is_violation)
                    # В файл
                    file_logger.log_speed(
                        frame_idx=yolo_result.frame_idx,
                        track_id=obj_id,
                        speed_kmh=current_speed,
                        plate_text=plate_text,
                    )

                # Сохраняем в кэш
                cached_detections.append({
                    'box': det.box,
                    'obj_id': obj_id,
                    'plate_text': plate_text,
                    'plate_conf': plate_conf,
                    'speed': current_speed,
                })

            # Cleanup
            if speed_method == "homography":
                speed.cleanup_old_tracks(yolo_result.frame_idx)

            # Performance log
            file_logger.log_performance(
                frame_idx=yolo_result.frame_idx,
                fps=metrics.current_fps,
                yolo_ms=yolo_result.processing_time_ms,
                ocr_queue=async_ocr.queue_size(),
                yolo_queue=async_yolo.input_queue.qsize(),
                cars_count=len(detections),
            )

            metrics.update_skip_reasons(recognizer.stats)
            metrics.end_frame()

        # === ПОЛУЧЕНИЕ OCR РЕЗУЛЬТАТОВ ===
        for ocr_result in async_ocr.get_results(current_frame_idx=frame_idx):
            if ocr_result.blur_score > 0:
                metrics.log_quality(ocr_result.blur_score, ocr_result.brightness_score)
            if ocr_result.result and ocr_result.result.plate_text:
                metrics.log_ocr(
                    called=True,
                    time_ms=ocr_result.processing_time_ms,
                    result=ocr_result.result.plate_text,
                    conf=ocr_result.result.ocr_conf,
                    passed=True,
                )

        # === ОТРИСОВКА (на КАЖДОМ кадре) ===
        raw_frame = frame.copy()

        # Зоны
        if speed_method == "lines" and hasattr(speed, 'draw_zones'):
            speed.draw_zones(raw_frame)

        # Рисуем детекции из кэша
        for det in cached_detections:
            x1i, y1i, x2i, y2i = det['box']
            obj_id = det['obj_id']
            plate_text = det['plate_text']
            plate_conf = det['plate_conf']
            current_speed = det['speed']

            box_color = (0, 255, 0) if plate_conf >= ocr_conf_threshold else (0, 255, 255)
            cv2.rectangle(raw_frame, (x1i, y1i), (x2i, y2i), box_color, 2)
            cv2.putText(raw_frame, f"ID:{obj_id}", (x1i, y1i - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if plate_text:
                cv2.putText(raw_frame, f"{plate_text} ({plate_conf:.2f})",
                            (x1i, y2i + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if current_speed:
                c = (0, 255, 0) if current_speed < speed_limit else (0, 0, 255)
                cv2.putText(raw_frame, f"{current_speed:.0f} km/h", (x1i, y1i - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

        if show_window:
            # Панель с результатами
            if recognizer.passed_results:
                raw_frame = draw_results_panel(
                    raw_frame, recognizer.passed_results,
                    speed_logger, speed_limit
                )

            # Bird eye view
            if show_bird_eye and speed_method == "homography":
                bev = speed.draw_bird_eye_view(size=(300, 400), frame_idx=frame_idx)
                bev_h, bev_w = bev.shape[:2]
                raw_frame[10:10+bev_h, raw_frame.shape[1]-bev_w-10:raw_frame.shape[1]-10] = bev

            cv2.putText(raw_frame, f"FPS: {metrics.current_fps:.1f}", (290, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Запись видео
            if video_writer is None:
                h, w = raw_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = AsyncVideoWriter(video_out_path, fourcc, fps, (w, h))
                print(f"🎬 Запись: {video_out_path} (async, {fps:.0f} fps)")
            video_writer.write(raw_frame)

            cv2.imshow(name, raw_frame)

            # === КОНТРОЛЬ FPS ПОКАЗА ===
            elapsed = time.time() - frame_start_time
            wait_ms = max(1, int((target_frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    # Завершение
    source.release()
    cv2.destroyAllWindows()

    # Закрываем видео
    if video_writer is not None:
        video_writer.release()
        print(f"🎬 Видео сохранено: {video_out_path}")

    # Останавливаем асинхронные модули
    async_yolo.stop()
    async_ocr.stop()

    # Сохранение результатов
    print("\n")  # новая строка после live-line
    yolo_stats = async_yolo.get_stats()
    print(f"📊 YOLO: submitted={yolo_stats['submitted']}, processed={yolo_stats['processed']}, dropped={yolo_stats['dropped']}")
    async_ocr.print_stats()
    recognizer.finalize()
    speed_logger.finalize()
    metrics.finalize()

    print(f"\n✅ Готово: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speed + ANPR (NomeroffNet)")
    parser.add_argument("--source", required=True, choices=["rtsp", "video", "folder", "image"])
    parser.add_argument("--camera", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    cfg = load_yaml(os.path.join(BASE_DIR, "config", "config.yaml"))
    cam_cfg = load_yaml(os.path.join(BASE_DIR, "config", "config_cam.yaml"))

    print_gpu_info()

    if args.source == "rtsp":
        if not args.camera:
            sys.exit("❗ Укажите --camera")
        source = create_source_from_config(cam_cfg, args.camera, prefer_hw=True)
    else:
        if not args.path:
            sys.exit("❗ Укажите --path")
        source_type = {
            "video": SourceType.VIDEO,
            "folder": SourceType.FOLDER,
            "image": SourceType.IMAGE,
        }[args.source]
        source = VideoSource(args.path, source_type=source_type, prefer_hw=True)

    process_source(source, cfg, cam_cfg)
