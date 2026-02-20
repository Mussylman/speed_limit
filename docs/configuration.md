# Конфигурация

Система использует два YAML-файла конфигурации и per-camera файлы гомографии.

## config/config.yaml — Глобальные настройки

Основной конфигурационный файл. Параметры можно переопределить в `config_cam.yaml` для каждой камеры.

### Общие параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `output_dir` | string | `"outputs"` | Корневая папка для результатов |
| `show_window` | bool | `true` | Показывать окно с видео |
| `fps` | int | `30` | FPS видеопотока (для расчёта скорости) |
| `device` | string | `"cuda"` | Устройство (`cuda` или `cpu`) |
| `frame_skip` | int | `3` | Обрабатывать каждый N-й кадр (YOLO) |

### Местоположение камеры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `address` | string | Адрес установки камеры |
| `coordinates` | [float, float] | GPS-координаты [lat, lon] |

### YOLO

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `models.yolo_model` | string | `"models/yolo11n.pt"` | Путь к YOLO модели |
| `yolo_imgsz` | int | `960` | Размер входа YOLO (640/960/1280) |

### Скорость

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `speed_method` | string | `"homography"` | Метод: `homography` или `lines` |
| `speed_limit` | int | `70` | Лимит скорости (км/ч) |

### Гомография

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `homography.config_file` | string | | Путь к YAML гомографии |
| `homography.min_track_points` | int | `5` | Мин. точек трека для расчёта |
| `homography.smoothing_window` | int | `10` | Окно сглаживания (кадры) |
| `homography.max_speed_kmh` | float | `200` | Макс. RAW скорость (фильтр) |
| `homography.min_speed_kmh` | float | `5` | Мин. RAW скорость (фильтр) |
| `homography.show_bird_eye` | bool | `true` | Показывать BEV-миникарту |
| `homography.speed_correction` | float | `0.43` | Поправочный коэффициент |

### OCR-фильтры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `min_car_height` | int | `40` | Мин. высота bbox машины (px) |
| `min_car_width` | int | `30` | Мин. ширина bbox машины (px) |
| `min_plate_width` | int | `85` | Мин. ширина номерной пластины (px) |
| `min_plate_height` | int | `10` | Мин. высота номерной пластины (px) |
| `min_plate_chars` | int | `7` | Мин. символов в номере |
| `ocr_conf_threshold` | float | `0.5` | Мин. уверенность OCR |
| `plate_format_regex` | string | `"^[0-9]{3}[A-Z]{2,3}[0-9]{2}$"` | Regex формата KZ номера |
| `ocr_cooldown_frames` | int | `1` | Интервал между OCR на одном треке |

### Линейные зоны (метод `lines`)

```yaml
line_zones:
  - name: "Down"
    start_line: [[903, 133], [1346, 170]]
    end_line: [[183, 504], [651, 643]]
    distance_m: 10
    direction: "down"
    color: [0, 255, 0]
```

---

## config/config_cam.yaml — Настройки камер

Per-camera конфигурация. Переопределяет глобальные параметры из `config.yaml`.

### Структура

```yaml
camera_credentials:
  user: admin
  password: ****
  port: 554
  stream_path: /stream1

cameras:
  - ip: 10.223.50.12
    name: Camera_12
    label: INNOVA-001
    # ...per-camera настройки...

cross_camera:
  enabled: true
  speed_limit: 70
  distances_m:
    - cameras: [Camera_12, Camera_14]
      distance: 41.3
    - cameras: [Camera_14, Camera_21]
      distance: 30.05
```

### Параметры камеры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `ip` | string | IP-адрес камеры |
| `name` | string | Имя камеры (используется в выводе и для поиска видеофайлов) |
| `label` | string | Лейбл (для отчётов) |
| `homography_config` | string | Путь к YAML гомографии для этой камеры |
| `speed_correction` | float | Поправочный коэффициент скорости |
| `speed_limit` | int | Лимит скорости (км/ч) |
| `min_car_height` | int | Мин. высота bbox машины |
| `min_car_width` | int | Мин. ширина bbox машины |
| `min_plate_width` | int | Мин. ширина номерной пластины |
| `min_confirmations` | int | Мин. подтверждений OCR (голосование) |
| `max_upgrades` | int | Макс. upgrade-попыток OCR на трек |
| `frame_skip` | int | Обработка каждого N-го кадра |
| `min_aspect_ratio` | float | Мин. соотношение сторон кропа (ширина/высота) |
| `max_aspect_ratio` | float | Макс. соотношение сторон кропа |
| `min_crop_width` | int | Мин. ширина кропа машины (px) |
| `min_crop_area` | int | Мин. площадь кропа (px^2) |
| `plate_stretch_x` | float | Горизонтальное растяжение номера (для re-OCR) |
| `ocr_resize_width` | int | Ширина resize кропа перед OCR (0 = без resize) |
| `cross_time_offset` | float | Компенсация задержки камеры (секунды) |

### Crop zone

Полигон, внутри которого запускается OCR. Задаётся как массив точек [x, y]:

```yaml
crop_zone:
  - [644, 422]
  - [1342, 364]
  - [2522, 1174]
  - [1004, 1418]
```

Настраивается через `tools/draw_zones.py`.

### Линейные зоны (per-camera)

Переопределяют глобальные `line_zones`:

```yaml
line_zones:
  - name: Down
    start_line: [[350, 420], [550, 420]]
    end_line: [[320, 480], [580, 480]]
    distance_m: 25
    direction: down
    color: [0, 255, 0]
```

### Кросс-камера

```yaml
cross_camera:
  enabled: true
  speed_limit: 70
  distances_m:
    - cameras: [Camera_12, Camera_14]
      distance: 41.3        # метры между Camera_12 и Camera_14
    - cameras: [Camera_14, Camera_21]
      distance: 30.05       # метры между Camera_14 и Camera_21
    # Camera_12 → Camera_21 = 41.3 + 30.05 = 71.35 (авто-сумма)
```

---

## Файлы гомографии

Для каждой камеры создаётся файл гомографии через `tools/calibrate_homography.py`:

```yaml
# config/homography_Camera_12.yaml
homography:
  source_points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  real_width_m: 10.0
  real_height_m: 25.0
  scale_px_per_m: 100.0
  matrix: [[...], [...], [...]]
```

Матрица также сохраняется отдельно в `.npy` файле.

---

## Quality presets

CLI-аргумент `--quality` меняет поведение системы:

### `--quality default` (live-режим)

Стандартные настройки из `config.yaml`. Подходит для RTSP-потоков в реальном времени:
- Пропуск кадров при переполнении очереди
- OCR cooldown между попытками
- Фильтры качества (blur, brightness)

### `--quality max` (offline-режим)

Убирает все ограничения реального времени. Подходит для обработки записанных видеофайлов:

| Параметр | default | max |
|----------|---------|-----|
| No-drop | `false` | `true` — ждём каждый кадр |
| OCR cooldown | `1` | `0` — OCR на каждом кадре |
| Quality filters | Включены | Отключены |
| Min confirmations | `2` | `2` |
| OCR scales | `6` | `6` (все масштабы + plate-zoom) |
| Prefetch | `8` | `16` |
| YOLO min conf | `0.5` | `0.25` |
