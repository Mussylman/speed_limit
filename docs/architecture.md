# Архитектура

## Общая схема пайплайна

```
VideoSource (RTSP / файл)
    │
    ▼
MotionDetector ── нет движения → пропуск кадра
    │
    ▼
AsyncYOLO (YOLO11 + ByteTrack)
    │
    ├── bbox + track_id + crop
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ Speed        │     │ OCR              │
│ Homography   │     │ (realtime или    │
│ (per-camera) │     │  deferred)       │
└──────┬──────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
  SpeedTracker          PlateRecognizer
       │                      │
       ├──────────┬───────────┤
       ▼          ▼           ▼
  speeds/    passed/failed   report/
  JSONL      res_ocr/       summary
```

## Компоненты

### VideoSource (`video/source.py`)

Абстракция видеоисточника: RTSP, файл, папка, изображение.

- **Prefetch**: отдельный поток декодирования, буферная очередь
- **NVDEC**: аппаратное декодирование через FFmpeg (RTSP)
- **No-drop**: `--quality max` ждёт каждый кадр без пропуска
- **Авто-реконнект**: при обрыве RTSP (WiFi нестабильность)

### AsyncYOLO / SharedAsyncYOLO (`async_yolo.py`, `shared_yolo.py`)

Детекция автомобилей в отдельном потоке.

**AsyncYOLO** (одна камера):
- 1 поток, 1 YOLO модель
- Вход: кадр → Выход: список детекций (bbox, track_id, conf, crop)
- ByteTrack для трекинга

**SharedAsyncYOLO** (мульти-камера):
- 1 YOLO модель на GPU, 1 worker-поток
- Несколько камер через `CameraYOLOProxy`
- Per-camera ByteTrack state (трекер переключается между камерами)
- Drop policy: при переполнении новый кадр вытесняет старый

```
Camera_12 ──┐
Camera_14 ──┤── SharedAsyncYOLO ── GPU ── результаты по камерам
Camera_21 ──┘   (1 модель)
```

### MotionDetector (`motion_detector.py`)

Лёгкий (~0.3ms) фильтр: пропускает YOLO для статичных кадров.

- Downscale до 320px → grayscale → absdiff с предыдущим → threshold
- Если движения нет → YOLO не вызывается (экономия GPU)
- Warmup: первые 5 кадров всегда обрабатываются

### PlateRecognizer (`plate_recognizer.py`)

OCR номерных знаков через NomeroffNet.

**Pipeline одного вызова:**
1. Кроп машины → resize до 800px → NomeroffNet (YOLO plate + OCR text)
2. Если OCR conf < 0.8 → **Smart Plate Re-OCR**:
   - Bbox номера из шага 1 → вырезаем plate region из оригинального hi-res кропа
   - Stretch по X (`plate_stretch_x`) + resize до 128px высоты → повторный OCR
3. Посимвольное слияние (`merge_texts_charwise`) двух результатов
4. Позиционная коррекция KZ-формата (`fix_kz_plate`)

**Фильтры (от дешёвого к дорогому):**
1. Размер машины (min_car_height/width)
2. Качество кадра (blur, brightness) — ~0.1ms
3. Улучшение vs предыдущий лучший (quality_improvement)
4. Cooldown (min интервал между OCR)
5. No-plate cooldown (пауза после N неудачных попыток)
6. → Запуск NomeroffNet
7. OCR confidence
8. Длина текста (min_plate_chars)
9. Формат regex (`^[0-9]{3}[A-Z]{2,3}[0-9]{2}$`)
10. Мульти-кадровое голосование (min_confirmations)

**Три score (0-1):**
- `car_score` — уверенность YOLO
- `plate_score` — качество кропа (blur + brightness + size)
- `ocr_score` — качество текста (длина + формат)
- `total_score` — среднее трёх

### AsyncOCR (`async_ocr.py`)

Асинхронная обёртка PlateRecognizer.

- Отдельный worker-поток, очередь задач
- **Дедуп**: max 1 задача на track_id в очереди
- **Upgrade**: если новый кроп крупнее предыдущего → повторный OCR
- **Max upgrades**: лимит на количество re-OCR для одного трека
- **Weighted voting**: текст с наибольшим весом побеждает (формат-валидный текст весит больше)
- **Vote-promote**: если трек не набрал min_confirmations в PlateRecognizer, но имеет достаточно weighted votes → promoted в passed

### SpeedHomography (`speed_homography.py`)

Измерение скорости через гомографию (перспективная проекция → метры).

**Алгоритм:**
1. Пиксель (cx, cy) → матрица гомографии → координаты в метрах (xm, ym)
2. Трек = deque точек `(frame, xm, ym)` за скользящее окно
3. Net displacement (первая → последняя точка) / время → RAW км/ч
4. Фильтры: min/max RAW скорость, jump detection (ID swap)
5. `speed_correction` — поправочный коэффициент, применяется к итогу
6. Медиана всех измерений = стабильная скорость для отчёта

**Bird's Eye View**: визуализация треков в метрах на миникарте.

### CrossCameraTracker (`cross_camera_speed.py`)

Средняя скорость по двум камерам.

**Принцип:**
- OCR распознаёт номер на Camera_A (время T1) и Camera_B (время T2)
- Средняя скорость = расстояние(A, B) / (T2 - T1)
- Расстояния заданы в `config_cam.yaml` → `cross_camera.distances_m`
- Непрямые расстояния суммируются автоматически (A→C = A→B + B→C)

**Реализация:**
- Daemon-поток, читает события из Queue
- `cross_time_offset` — компенсация задержки камеры (секунды)
- TTL: сброс sightings через 5 минут
- Дедуп: одна пара (plate, cam_from, cam_to) считается один раз

## Два режима OCR

### Realtime (прямой)

Используется при `--source rtsp` с одной камерой или `--no-defer`:

```
Кадр → YOLO → crop → AsyncOCR → PlateRecognizer → NomeroffNet → результат
                                     (1 GPU-worker)
```

- OCR работает параллельно с детекцией
- Ограничение: `_ocr_max_scales: 2` (быстрее, но менее точно)
- Номер появляется в OSD сразу после распознавания

### Deferred (пост-обработка)

Используется по умолчанию для `--source rtsp` с несколькими камерами:

```
Фаза 1 (live):   Кадр → YOLO → crop → CropCollector → disk
Фаза 2 (post):   Кропы с диска → NomeroffNet → результаты
```

- Во время live-фазы GPU занят только YOLO (максимальный FPS трекинга)
- CropCollector сохраняет ВСЕ кропы на диск + лучшие K в памяти
- После завершения live-фазы: загрузка NomeroffNet, OCR на лучших кропах
- `_ocr_max_scales: 6` (все масштабы + plate-zoom → лучшее качество)

## SharedYOLO (мульти-камера)

```
                          ┌── Camera_12 proxy
SharedAsyncYOLO ──────────┤── Camera_14 proxy
  (1 YOLO model)          └── Camera_21 proxy
  (1 GPU worker)

Внутри worker:
  1. Берёт задачу из общей очереди
  2. Переключает ByteTrack state на камеру задачи
  3. YOLO inference
  4. Кладёт результат в per-camera output queue
```

**CameraYOLOProxy** — drop-in замена AsyncYOLO для main.py:
- Тот же API (submit, get_results, get_stats)
- Маршрутизирует через SharedAsyncYOLO

## KZ Plate коррекция (`kz_plate.py`)

Казахстанский формат: `XXX YYY XX` (3 цифры + 2-3 буквы + 2 цифры региона).

**Позиционная коррекция:**
- Позиции 0-2: буква → цифра (O→0, I→1, B→8, S→5, G→6, T→7)
- Позиции 3-5: цифра → буква (0→O, 1→I, 8→B, 5→S, 6→G, 7→T)
- Позиции 6-7: буква → цифра

**Восстановление (7 символов):**
- Пробует вставить пропущенный символ в каждую позицию
- Бонус за валидный регион (01-21)

**Посимвольное голосование (`merge_texts_charwise`):**
- Несколько OCR-чтений → мажоритарный выбор на каждой позиции
- Группировка визуально похожих символов (A/Z/C → одна группа)

**Fuzzy matching (`match_plate_fuzzy`):**
- Проверяет варианты OCR-путаниц (A↔Z, H↔N) против уже известных номеров
- Исправляет ошибки без повторного OCR
