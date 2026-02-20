# Korgen Vision v1.0

Система фиксации скорости и распознавания номерных знаков на дорогах Казахстана.

3 камеры, YOLO-детекция автомобилей, NomeroffNet OCR номерных знаков, измерение скорости через гомографию и кросс-камерный замер средней скорости.

## Возможности

- **Детекция автомобилей** — YOLO11 + ByteTrack (GPU, асинхронный pipeline)
- **Распознавание номеров** — NomeroffNet (TensorRT) с коррекцией KZ-формата
- **Измерение скорости** — гомография (bird's eye view) для каждой камеры
- **Кросс-камерная скорость** — средняя скорость между камерами по расстоянию и времени
- **3 камеры одновременно** — SharedYOLO + потоковая обработка на 1 GPU
- **Два режима OCR** — realtime (прямой) и deferred (пост-обработка)
- **Отчёты** — карточки нарушителей, summary, JSON, видеозаписи

## Требования

| Компонент       | Минимум                          |
|-----------------|----------------------------------|
| GPU             | NVIDIA RTX 3050 (4 GB) или выше  |
| Python          | 3.10                             |
| CUDA            | 11.8+                            |
| FFmpeg          | 6.0+ (с NVENC для GPU-записи)   |
| ОС              | Windows 10/11, Linux             |

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/Mussylman/korgen_vision_speed_limit.git speed_limit
cd speed_limit

# 2. Установить зависимости
pip install ultralytics torch torchvision opencv-python-headless pyyaml pillow

# 3. Клонировать NomeroffNet (локально, не pip)
git clone https://github.com/ria-com/nomeroff-net.git nomeroff-net

# 4. Скачать/подготовить модели
#    - YOLO: models/yolo11n.pt (скачивается автоматически)
#    - NomeroffNet TensorRT: nomeroff-net/data/models/Detector/yolov11x/yolov11x-keypoints-2024-10-11.engine

# 5. Настроить камеры
#    Отредактировать config/config.yaml и config/config_cam.yaml

# 6. Запуск (одна камера, видеофайл)
python src/cli.py --source video --path video.mp4 --camera Camera_12 --quality max

# 7. Запуск (три камеры, видео)
python src/cli.py --source video --camera Camera_12 Camera_14 Camera_21 \
    --path "records/test/" --quality max --name "test_60kmh"
```

Подробнее: [docs/installation.md](docs/installation.md)

## Структура проекта

```
speed_limit/
├── src/                          # Основной код
│   ├── cli.py                    # CLI точка входа
│   ├── main.py                   # Главный pipeline (YOLO + Speed + OCR)
│   ├── config.py                 # Загрузка конфигов, пути
│   ├── pipeline_builder.py       # Сборка компонентов (YOLO, OCR, Speed)
│   ├── plate_recognizer.py       # OCR номеров через NomeroffNet
│   ├── async_ocr.py              # Асинхронный OCR (очередь, upgrade, voting)
│   ├── kz_plate.py               # Коррекция KZ номеров (формат, путаницы)
│   ├── speed_homography.py       # Скорость через гомографию
│   ├── speed_line.py             # Скорость через линии (альтернатива)
│   ├── speed_tracker.py          # Трекинг скорости по vehicle ID
│   ├── cross_camera_speed.py     # Кросс-камерный замер средней скорости
│   ├── shared_yolo.py            # SharedAsyncYOLO для мульти-камер
│   ├── async_yolo.py             # Асинхронный YOLO (одна камера)
│   ├── crop_collector.py         # Сбор кропов (deferred OCR)
│   ├── motion_detector.py        # Детектор движения (skip static)
│   ├── report_generator.py       # Генератор отчётов
│   ├── file_logger.py            # Файловый логгер
│   ├── metrics_logger.py         # Метрики (FPS, YOLO, OCR)
│   └── video/
│       ├── source.py             # VideoSource (RTSP, файл, папка)
│       ├── writer.py             # AsyncVideoWriter (FFmpeg)
│       └── decoder.py            # Аппаратный декодер
├── config/
│   ├── config.yaml               # Глобальные настройки
│   ├── config_cam.yaml           # Настройки камер
│   └── homography_*.yaml         # Матрицы гомографии
├── tools/                        # Утилиты
│   ├── calibrate_homography.py   # Калибровка гомографии
│   ├── estimate_distance.py      # Проверка расстояний
│   ├── record_stream.py          # Запись RTSP потока
│   ├── grab_frame.py             # Захват кадра с камеры
│   ├── draw_zones.py             # Рисование crop_zone
│   ├── batch_ocr.py              # Тестирование OCR на кропах
│   ├── grid_live.py              # Мониторинг 3 камер
│   └── cross_camera_report.py    # Генерация кросс-отчёта
├── models/                       # YOLO модели
├── nomeroff-net/                 # NomeroffNet (локальный клон)
├── records/                      # Записи RTSP
└── outputs/                      # Результаты (exp_N/)
```

## Документация

- [Установка](docs/installation.md)
- [Конфигурация](docs/configuration.md)
- [Использование](docs/usage.md)
- [Архитектура](docs/architecture.md)
- [Выходные данные](docs/output.md)
- [Утилиты](docs/tools.md)
- [Калибровка](docs/calibration.md)
