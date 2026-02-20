# Выходные данные

Каждый запуск создаёт директорию `outputs/exp_N/`. При мульти-камерном режиме внутри — поддиректории по камерам.

## Структура outputs/exp_N/

### Одна камера

```
outputs/exp_1/
└── Camera_12/
    ├── passed/                    # Прошли все фильтры OCR
    │   ├── images/                # Кропы машин (JPEG 95%)
    │   │   ├── a1b2c3d4_123ABC01.jpg
    │   │   └── ...
    │   └── results.json           # Массив PlateEvent
    │
    ├── failed/                    # Лучшие из не прошедших
    │   ├── images/                # Кропы
    │   └── results.json           # Массив PlateEvent + reject_reason
    │
    ├── res_ocr/                   # Crash-safe немедленное сохранение
    │   ├── <timestamp>_<plate>_id<track>.jpg     # Кроп машины
    │   ├── <timestamp>_<plate>_id<track>_full.jpg # Полный кадр
    │   └── <timestamp>_<plate>_id<track>.json    # PlateEvent
    │
    ├── debug_ocr/                 # Полная отладка OCR
    │   ├── car_crops/             # Все кропы, отправленные на OCR
    │   ├── plate_bbox/            # Кропы с bbox номера
    │   ├── plate_crops/           # Вырезанные номерные пластины
    │   ├── no_plate/              # Номер не найден
    │   ├── small_plate/           # Номер слишком маленький
    │   └── rejected/              # Отклонённые (размер/качество)
    │
    ├── speeds/                    # Данные о скорости
    │   ├── measurements.jsonl     # Потоковые измерения
    │   ├── all_speeds.json        # Итоговая сводка
    │   └── violations.json        # Только нарушители
    │
    ├── report/                    # Клиентский отчёт
    │   ├── vehicles/              # Карточки машин (JPG)
    │   │   ├── 01_123ABC01_85kmh_VIOLATION.jpg
    │   │   └── 02_456DEF02_55kmh.jpg
    │   ├── violations/            # Только нарушители
    │   ├── summary.jpg            # Обзорное изображение
    │   ├── report.txt             # Текстовый отчёт
    │   └── video.mp4              # Обработанное видео
    │
    ├── detections.jsonl           # Лог YOLO-детекций
    ├── metrics.jsonl              # Метрики по кадрам
    └── metrics_summary.json       # Сводка метрик
```

### Три камеры

```
outputs/exp_2/
├── Camera_12/                     # Результаты камеры 12
│   ├── passed/
│   ├── failed/
│   ├── res_ocr/
│   ├── speeds/
│   └── report/
├── Camera_14/                     # Результаты камеры 14
│   └── ...
├── Camera_21/                     # Результаты камеры 21
│   └── ...
├── cross_camera_speeds.json       # Кросс-камерный отчёт (JSON)
└── cross_camera_report.jpg        # Кросс-камерный отчёт (изображение)
```

---

## Описание файлов

### passed/results.json

Массив `PlateEvent` — номера, прошедшие все фильтры:

```json
[
  {
    "event_id": "a1b2c3d4",
    "timestamp": "2026-02-18T13:32:15.123456",
    "camera_id": "Camera_12",
    "camera_label": "INNOVA-001",
    "track_id": 42,
    "frame_idx": 1250,
    "car_score": 0.95,
    "plate_score": 0.82,
    "ocr_score": 0.90,
    "total_score": 0.89,
    "detection_conf": 0.95,
    "plate_conf": 0.88,
    "ocr_conf": 0.76,
    "text_conf": 0.80,
    "plate_text": "123ABC01",
    "region": "kz",
    "brightness": 125.3,
    "blur": 18500.0,
    "plate_width_px": 95,
    "plate_height_px": 28,
    "car_width_px": 480,
    "car_height_px": 350,
    "crop_path": "images/a1b2c3d4_123ABC01.jpg",
    "reject_reason": ""
  }
]
```

### failed/results.json

Тот же формат, но с заполненным `reject_reason`:
- `"low_conf:0.35"` — низкая уверенность OCR
- `"chars:6"` — мало символов
- `"format:123AB1"` — не соответствует regex

### res_ocr/

Немедленное сохранение при распознавании (переживает Ctrl+C / crash):

- `<timestamp>_<plate>_id<track>.jpg` — кроп машины
- `<timestamp>_<plate>_id<track>_full.jpg` — полный кадр с камеры
- `<timestamp>_<plate>_id<track>.json` — PlateEvent (JSON)

Полезно когда основной `passed/results.json` не успел записаться.

### debug_ocr/

Полная отладочная информация OCR (для анализа ошибок распознавания):

| Подпапка | Содержимое |
|----------|-----------|
| `car_crops/` | Все кропы машин, отправленные на OCR (с оригинальным размером) |
| `plate_bbox/` | Кропы с нарисованным bbox номерной пластины |
| `plate_crops/` | Вырезанные номерные пластины |
| `no_plate/` | Кропы где NomeroffNet не нашёл номер |
| `small_plate/` | Кропы где номер слишком маленький |
| `rejected/` | Отклонённые фильтрами (размер/качество) |

Формат имён файлов: `{idx}_f{frame}_t{track}_s{scale}_{text}.jpg`

### speeds/measurements.jsonl

Потоковый лог измерений скорости (каждая строка — JSON):

```json
{"ts": "2026-02-18T13:32:15", "frame": 1250, "track_id": 42, "speed": 72.3, "plate": "123ABC01"}
```

### speeds/all_speeds.json

Итоговая сводка по всем машинам:

```json
{
  "camera_id": "Camera_12",
  "speed_limit": 70,
  "stats": {
    "total_measurements": 4500,
    "unique_vehicles": 45,
    "violations": 5,
    "max_speed": 92.1,
    "avg_speed": 58.4
  },
  "vehicles": [
    {
      "track_id": 42,
      "speed_kmh": 72.3,
      "is_violation": true,
      "plate_text": "123ABC01",
      "plate_conf": 0.76,
      "first_frame": 1100,
      "last_frame": 1400,
      "measurement_count": 28
    }
  ]
}
```

### speeds/violations.json

Только нарушители (> speed_limit):

```json
{
  "speed_limit": 70,
  "count": 5,
  "violations": [...]
}
```

### report/

Клиентский отчёт для презентации:

- **vehicles/** — карточки машин: кроп + номер + скорость (красная рамка для нарушителей)
- **violations/** — только нарушители
- **summary.jpg** — обзорное изображение со статистикой и миниатюрами
- **report.txt** — текстовый отчёт с таблицами
- **video.mp4** — обработанное видео (с bbox, скоростями, номерами)

### cross_camera_speeds.json

Кросс-камерный отчёт (при мульти-камерном режиме):

```json
{
  "name": "60 km/h test",
  "speed_limit": 70,
  "results": [
    {
      "plate": "123ABC01",
      "cam_from": "Camera_12",
      "cam_to": "Camera_14",
      "distance_m": 41.3,
      "time_sec": 2.8,
      "speed_kmh": 53.1
    }
  ]
}
```

### cross_camera_report.jpg

Визуальный отчёт: карточки совпавших номеров + таблица скоростей + маркировка нарушителей.
