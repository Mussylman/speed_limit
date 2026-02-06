# Speed Limit — система измерения скорости и распознавания номеров

Система видеоаналитики для измерения скорости транспортных средств, распознавания номерных знаков (ANPR) и фиксации нарушений ПДД. Включает Python-бэкенд с GPU-ускорением и React-фронтенд для управления и мониторинга.

## Архитектура

```
┌─────────────────────────────────────────────────┐
│           React Frontend (порт 5173)            │
│   TypeScript · Zustand · TailwindCSS · Leaflet  │
└──────────────┬──────────────────┬───────────────┘
          REST API           WebSocket
               │                  │
┌──────────────▼──────────────────▼───────────────┐
│          FastAPI Backend (порт 8000)             │
│   REST endpoints · MJPEG · WebSocket events      │
└──────────────┬──────────────────────────────────┘
               │
   ┌───────────┼───────────────┐
   │           │               │
 YOLO11    NomeroffNet    Speed Estimation
 детекция    OCR номеров   (гомография/линии)
 +ByteTrack
```

## Возможности

**Видеопайплайн:**
- Детекция и трекинг автомобилей (YOLO11 + ByteTrack)
- Измерение скорости (гомография или линейный метод)
- Распознавание номеров (NomeroffNet, казахские номера)
- Асинхронный OCR для максимального FPS
- Логирование нарушений скоростного режима

**API-сервер:**
- REST API для камер, нарушений, скоростей, номеров
- WebSocket для реального времени (скорость, нарушения, номера)
- MJPEG стриминг и снапшоты
- Swagger/ReDoc документация

**Фронтенд:**
- Дашборд мониторинга системы
- Управление камерами (CRUD, запуск/остановка)
- Интерактивная карта камер (Leaflet)
- Список нарушений с фильтрацией
- Отслеживание маршрутов по номерам
- Живое видео (HLS.js)
- Мультиязычность (русский, казахский, английский)

## Структура проекта

```
speed_limit/
├── src/                        # Python — видеопайплайн
│   ├── main.py                 # Главный pipeline
│   ├── async_yolo.py           # Асинхронная YOLO детекция
│   ├── async_ocr.py            # Асинхронный OCR
│   ├── plate_recognizer.py     # Распознавание номеров
│   ├── speed_homography.py     # Скорость через гомографию
│   ├── speed_line.py           # Скорость через линии
│   ├── speed_logger.py         # Логирование нарушений
│   ├── metrics_logger.py       # Метрики производительности
│   ├── file_logger.py          # Файловый логгер
│   ├── video/                  # Источники видео
│   │   ├── source.py           # RTSP / файл / папка + prefetch
│   │   └── decoder.py          # NVDEC → GStreamer → FFmpeg
│   └── api/                    # FastAPI REST + WebSocket
│       ├── server.py           # Endpoints
│       ├── websocket.py        # WS manager + topics
│       ├── schemas.py          # Pydantic-схемы
│       ├── storage.py          # Чтение outputs/ + кэш
│       ├── mjpeg.py            # MJPEG preview
│       └── integration.py      # Pipeline → API events
│
├── sergek-frontend/            # React 19 + TypeScript
│   ├── src/
│   │   ├── pages/              # Страницы (Dashboard, Cameras, Map, ...)
│   │   ├── components/         # UI-компоненты
│   │   ├── services/api/       # Axios-клиент, сервисы
│   │   ├── stores/             # Zustand state management
│   │   ├── i18n/               # Переводы (ru, kk, en)
│   │   ├── router/             # React Router
│   │   └── types/              # TypeScript типы
│   ├── package.json
│   └── vite.config.ts
│
├── config/
│   ├── config.yaml             # Основной конфиг
│   ├── config_cam.yaml         # Камеры и RTSP-доступ (не в git)
│   └── homography_config.yaml  # Калибровка гомографии
│
├── models/                     # ML-модели (не в git, скачать отдельно)
├── tools/                      # Утилиты калибровки
├── videos/                     # Тестовые видео (не в git)
├── outputs/                    # Результаты работы (генерируются)
├── records/                    # Записи видео (генерируются)
│
├── run_api.py                  # Запуск API-сервера
├── requirements.txt            # Python-зависимости (pipeline)
└── requirements_api.txt        # Python-зависимости (API)
```

## Требования

### Backend
- Python 3.10+
- CUDA 11.8+
- TensorRT 8.6+

### Frontend
- Node.js >= 18.0

## Быстрый старт

### 1. Клонирование

```bash
git clone https://github.com/Mussylman/speed_limit.git
cd speed_limit
```

### 2. Backend — установка

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux

pip install -r requirements.txt
pip install -r requirements_api.txt
```

### 3. Модели

Скачайте модели и поместите в `models/`:
- `yolo11n.engine` (или `.onnx` / `.pt`) — детекция
- `nomeroff_kz_ocr.engine` (или `.onnx`) — OCR номеров

### 4. Конфигурация камер

Создайте `config/config_cam.yaml` по образцу:

```yaml
camera_credentials:
  user: "admin"
  password: "your_password"
  port: 554
  stream_path: "/stream1"

cameras:
  - ip: "192.168.1.108"
    name: "Camera_108"
    line_zones:
      - name: "Down"
        start_line: [[640, 850], [1280, 850]]
        end_line: [[520, 550], [1280, 550]]
        distance_m: 32
        direction: "up"
```

### 5. Запуск видеопайплайна

```bash
# Из видеофайла
python src/main.py --source video --path videos/test.mp4

# Из RTSP-потока
python src/main.py --source rtsp --camera Camera_108
```

### 6. Запуск API-сервера

```bash
python run_api.py --port 8000

# С auto-reload (разработка)
python run_api.py --reload
```

Документация API: http://localhost:8000/docs

### 7. Запуск фронтенда

```bash
cd sergek-frontend
npm install
npm run dev
```

Фронтенд: http://localhost:5173

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/cameras` | Список камер |
| POST | `/api/cameras/{id}/start` | Запуск обработки |
| POST | `/api/cameras/{id}/stop` | Остановка |
| GET | `/api/violations` | Нарушения (с пагинацией) |
| GET | `/api/vehicles` | Распознанные номера |
| GET | `/api/speeds` | Измерения скорости |
| GET | `/api/stream/{id}/mjpeg` | MJPEG видеопоток |
| GET | `/api/stream/{id}/snapshot` | Текущий кадр |
| GET | `/api/metrics/{id}` | Метрики производительности |
| GET | `/api/health` | Статус системы |

### WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
ws.send(JSON.stringify({ action: "subscribe", channel: "speed:Camera_108" }));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

Каналы: `speed:{id}`, `violations:{id}`, `plates:{id}`, `system`

## Конфигурация (config/config.yaml)

```yaml
device: "cuda"
frame_skip: 2
yolo_imgsz: 640            # 640=быстро, 1280=качество
speed_method: "homography"  # или "lines"
speed_limit: 70             # км/ч
min_car_height: 80
min_car_width: 60
min_plate_chars: 8
ocr_conf_threshold: 0.5
```

## Производительность

| Компонент | Время |
|-----------|-------|
| YOLO детекция | 11-15 ms |
| Трекинг (ByteTrack) | 1-2 ms |
| OCR (NomeroffNet) | 25-30 ms |
| **Итого FPS** | **25-35** |

## Инструменты калибровки

```bash
python tools/calibrate_homography.py   # Настройка гомографии
python tools/draw_line_zones.py        # Настройка линейных зон
```

## Выходные данные

```
outputs/<camera>_run_<timestamp>/
├── passed/images/         # Кропы машин (распознано)
├── passed/results.json    # Номера
├── failed/images/         # Кропы (не распознано)
├── speeds/all_speeds.json # Все измерения
├── speeds/violations.json # Нарушители
├── metrics.jsonl          # Метрики по кадрам
└── metrics_summary.json   # Сводка
```

## Технологический стек

**Backend:** Python, FastAPI, YOLO11, ByteTrack, NomeroffNet, OpenCV, PyTorch, TensorRT, ONNX

**Frontend:** React 19, TypeScript, Vite, TailwindCSS, Zustand, React Router, Leaflet, HLS.js, Axios, i18next, Framer Motion
