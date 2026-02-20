# Использование

## CLI-аргументы

```
python src/cli.py --source <тип> [опции]
```

| Аргумент | Тип | Обязательный | Описание |
|----------|-----|:---:|----------|
| `--source` | `rtsp` / `video` / `folder` / `image` | Да | Тип источника видео |
| `--camera` | string (несколько) | Для RTSP | Имя камеры из `config_cam.yaml` |
| `--path` | string | Для video/folder/image | Путь к файлу/папке |
| `--quality` | `default` / `max` | Нет | Пресет качества |
| `--start-time` | string | Нет | Время начала видео: `DD-MM-YYYY HH:MM:SS` |
| `--no-defer` | flag | Нет | Мульти-камера: realtime OCR вместо deferred |
| `--name` | string | Нет | Имя теста (для кросс-камерного отчёта) |

## Режимы работы

### 1. Одна камера — видеофайл

Обработка записанного видео с максимальным качеством:

```bash
python src/cli.py \
    --source video \
    --path records/Camera_12/Camera_12_2026-02-18_11-56-38.mp4 \
    --camera Camera_12 \
    --quality max
```

- `--quality max` отключает дропы, cooldown, фильтры — обрабатывается каждый кадр
- `--camera Camera_12` применяет настройки из `config_cam.yaml` (speed_correction, min_plate_width и т.д.)
- Результат: `outputs/exp_N/Camera_12/`

### 2. Три камеры — видео (последовательно)

Обработка записей с трёх камер + кросс-камерный отчёт:

```bash
python src/cli.py \
    --source video \
    --camera Camera_12 Camera_14 Camera_21 \
    --path "records/test_60kmh/" \
    --quality max \
    --name "60 km/h test"
```

**Поиск видеофайлов:**
- Одна папка: ищет `Camera_12_*.mp4`, `Camera_14_*.mp4`, `Camera_21_*.mp4`
- Несколько папок (`;`-разделитель): `--path "dir1;dir2;dir3"`
- Несколько файлов (`;`-разделитель): `--path "cam12.mp4;cam14.mp4;cam21.mp4"`

**Процесс:**
1. Обрабатывает Camera_12, Camera_14, Camera_21 последовательно
2. Каждая камера: YOLO + Speed + OCR → результаты в `outputs/exp_N/Camera_XX/`
3. После всех камер: автоматический кросс-камерный отчёт
4. Результат: `outputs/exp_N/cross_camera_speeds.json` + `cross_camera_report.jpg`

### 3. RTSP live — три камеры (многопоточный)

Обработка RTSP-потоков в реальном времени:

```bash
python src/cli.py \
    --source rtsp \
    --camera Camera_12 Camera_14 Camera_21
```

**Архитектура:**
- Все 3 камеры работают параллельно в отдельных потоках
- Общий YOLO (SharedAsyncYOLO) — 1 модель на GPU
- OCR по умолчанию **deferred** (кропы собираются, OCR после завершения)
- Кросс-камера: CrossCameraTracker считает скорость в реальном времени

**Realtime OCR (вместо deferred):**
```bash
python src/cli.py \
    --source rtsp \
    --camera Camera_12 Camera_14 Camera_21 \
    --no-defer
```

- `--no-defer` загружает NomeroffNet сразу с YOLO (shared pipeline + lock)
- OCR идёт параллельно с детекцией, но с ограничением масштабов (`_ocr_max_scales: 2`)

### 4. Одна камера — RTSP live

```bash
python src/cli.py \
    --source rtsp \
    --camera Camera_12
```

- Одна камера, собственный YOLO + OCR
- `show_window: true` — показывает окно с bbox, скоростью, номерами
- Запись видео в `records/`

### 5. Папка изображений

```bash
python src/cli.py \
    --source folder \
    --path images/ \
    --camera Camera_12
```

### 6. Одно изображение

```bash
python src/cli.py \
    --source image \
    --path config/frame_Camera_12.jpg \
    --camera Camera_12
```

## Время начала видео

Для корректных timestamps в результатах (важно для кросс-камеры) указывайте время начала:

```bash
# Ручное указание
python src/cli.py --source video --path video.mp4 --camera Camera_12 \
    --start-time "18-02-2026 13:31:41"
```

**Автоопределение (приоритет):**
1. `--start-time` (CLI) — ручное
2. `meta.json` в папке видео (сохраняется `record_stream.py`)
3. Из имени файла (формат `Camera_12_2026-02-18_11-56-38.mp4`)
4. По mtime файла (наименее точный)

## Вывод в консоли

Каждые 10 секунд выводится блок статистики:

```
======================================================================
[STREAM] FPS:25.3 decode:30 queue:2/16 drop:0 null:0 reconnect:0 lat:45ms(p95:82)
[YOLO]   submit:1250 done:1248 drop:2(0%) time:14ms
[OCR]    called:340 no_plate:180 skip: size=50 blur=12 dup=45 cool=0 fmt=8 chars=3 plate_sz=20 low_conf=5 unconf=12
[OCR-Q]  submit:340 done:335 upgrades:28 drop: full=0 flight=5 max_upg=0 not_bigger=0
[FILTER] aspect_skip:15 width_skip:8 area_skip:3 (ratio:0.9-1.4 w>=350 area>=40000)
[MOTION] total:1250 motion:1100 static:150
[RESULT] cars:45 plates:32 violations:5
======================================================================
```

| Метрика | Описание |
|---------|----------|
| `FPS` | Частота обработки кадров |
| `decode` | FPS декодирования видеопотока |
| `queue` | Размер prefetch-очереди |
| `drop` | Пропущенные кадры (переполнение) |
| `lat` | Задержка capture→process (мс) |
| `p95` | 95-й перцентиль задержки |
| `cars` | Уникальные машины (с измеренной скоростью) |
| `plates` | Распознанные номера (прошли все фильтры) |
| `violations` | Нарушения (скорость > лимита) |

## Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| `Q` | Выход (сохраняет результаты) |
| `Ctrl+C` | Выход (сохраняет результаты) |
