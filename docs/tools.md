# Утилиты

Папка `tools/` содержит вспомогательные скрипты для калибровки, тестирования и мониторинга.

## calibrate_homography.py — Калибровка гомографии

Интерактивный инструмент для создания матрицы гомографии (перспектива → bird's eye view).

```bash
python tools/calibrate_homography.py \
    --video records/Camera_12/Camera_12_2026-02-18.mp4 \
    --camera Camera_12
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--video` | (обязательный) | Путь к видео или изображению |
| `--camera` | | Имя камеры (для имени выходного файла) |
| `--output` | `config/homography_{camera}.yaml` | Путь выходного YAML |
| `--max-width` | `1280` | Макс. ширина окна |

**Процесс:**
1. Открывается кадр из видео
2. Кликаем 4 точки дорожного полотна (полигон)
3. Вводим реальные размеры в метрах (ширина × длина)
4. Генерируется матрица гомографии

**Вывод:**
- `config/homography_Camera_12.yaml` — конфигурация
- `config/homography_Camera_12_matrix.npy` — матрица (NumPy)

Подробнее: [docs/calibration.md](calibration.md)

---

## estimate_distance.py — Проверка расстояний

Проверяет корректность гомографии: показывает пройденный путь каждой машины в метрах.

```bash
python tools/estimate_distance.py \
    records/Camera_12/video.mp4 \
    --homography config/homography_Camera_12.yaml \
    --correction 0.805
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `video` | (обязательный) | Путь к видео |
| `--homography` | `config/homography_config.yaml` | Путь к YAML гомографии |
| `--model` | `models/yolo11n.pt` | YOLO модель |
| `--imgsz` | `960` | Размер входа YOLO |
| `--correction` | `0.43` | Поправочный коэффициент |
| `--no-show` | | Без окна (headless) |

**Вывод:**
- Видео с наложенными расстояниями
- Карточки машин (JPG) со статистикой пути
- Таблица в консоли

---

## record_stream.py — Запись RTSP потока

Записывает RTSP-потоки в MP4 через FFmpeg (`-c copy`, без перекодирования).

```bash
# Одна камера
python tools/record_stream.py --camera Camera_12

# Три камеры + имя теста
python tools/record_stream.py \
    --camera Camera_12 Camera_14 Camera_21 \
    --name "test_60kmh"
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--camera` | (обязательный) | Имена камер (одна или несколько) |
| `--name` | | Имя теста (создаёт подпапку `records/{name}/`) |
| `--output` | `records/` | Папка для записей |

**Особенности:**
- Zero re-encoding: `ffmpeg -c copy` (максимальное качество)
- Auto-reconnect при обрыве WiFi
- `meta.json` рядом с каждым файлом (start_epoch для timestamps)
- Fragmented MP4 (`-movflags frag_keyframe`) — crash-safe
- Auto-remux после записи (фикс для некоторых плееров)

**Вывод:**
```
records/test_60kmh/
├── Camera_12_2026-02-18_13-30-00.mp4
├── Camera_12_2026-02-18_13-30-00_meta.json
├── Camera_14_2026-02-18_13-30-00.mp4
├── ...
```

---

## grab_frame.py — Захват кадра с камеры

Сохраняет один кадр из RTSP-потока (для калибровки, настройки зон).

```bash
python tools/grab_frame.py --camera Camera_12
python tools/grab_frame.py --camera Camera_12 --output config/frame_Camera_12.jpg
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--camera` | (обязательный) | Имя камеры |
| `--output` | `frame_{camera}.jpg` | Путь выходного JPEG |

---

## draw_zones.py — Рисование зон

Интерактивный GUI для рисования полигональных зон (crop_zone, detection_zone).

```bash
python tools/draw_zones.py \
    --camera Camera_12 \
    --image config/frame_Camera_12.jpg
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--camera` | (обязательный) | Имя камеры |
| `--image` / `--video` | (обязательный) | Источник кадра |
| `--zone` | `crop_zone` | Тип зоны (`crop_zone` или `detection_zone`) |
| `--width` | `1280` | Ширина окна |

**Управление:**
- ЛКМ — добавить точку
- ПКМ — отменить последнюю точку
- `R` — сбросить все точки
- `Enter` — сохранить в `config_cam.yaml`
- `Q` / `Esc` — выход без сохранения

---

## batch_ocr.py — Тестирование OCR на кропах

Запускает NomeroffNet на папке с кропами машин. Полезно для оценки качества OCR.

```bash
python tools/batch_ocr.py outputs/exp_1/Camera_12/debug_ocr/car_crops/
python tools/batch_ocr.py outputs/exp_1/Camera_12/debug_ocr/car_crops/ --resize 800
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `folder` | (обязательный) | Папка с кропами |
| `--resize` | `0` | Ширина resize перед OCR (0 = без resize) |
| `--csv` | `<folder>/batch_ocr.csv` | Путь выходного CSV |

**Вывод:**
- CSV: file, frame, track_id, crop_w, crop_h, text, conf_min, conf_avg, plate_w, det_conf
- Консоль: сводка по трекам (варианты OCR, уверенность)

---

## grid_live.py — Мониторинг 3 камер

Живой мониторинг трёх камер в сетке с YOLO-детекцией (без OCR — лёгкий режим).

```bash
# Авто-поиск последних видео
python tools/grid_live.py

# Ручные пути
python tools/grid_live.py --videos cam12.mp4 cam14.mp4 cam21.mp4

# Настройки
python tools/grid_live.py --speed 2.0 --imgsz 640 --skip 2
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--videos` | авто-поиск | 3 видеопути |
| `--speed` | `1.0` | Множитель скорости воспроизведения |
| `--imgsz` | `960` | Размер входа YOLO |
| `--skip` | `1` | Каждый N-й кадр |
| `--max-detect-size` | `0` | Resize перед YOLO (0 = без resize) |

**Управление:**
- `SPACE` — пауза/возобновление
- `Q` — выход
- `+` / `-` — скорость воспроизведения
- `D` — перемотка +30 кадров
- `A` — перемотка -30 кадров

---

## cross_camera_report.py — Кросс-камерный отчёт

Генерация кросс-камерного отчёта из результатов обработки нескольких камер.

```bash
# Из experiment directory (авто-поиск Camera_* подпапок)
python tools/cross_camera_report.py --exp outputs/exp_1/

# Явные директории
python tools/cross_camera_report.py \
    --dirs outputs/exp_1/Camera_12 outputs/exp_1/Camera_14 outputs/exp_1/Camera_21 \
    --name "Test 60 km/h"
```

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--exp` | | Корневая директория эксперимента |
| `--dirs` | | Явные пути к Camera_* директориям |
| `--name` | | Название теста (заголовок отчёта) |
| `--output` | | Путь выходного изображения |

**Сопоставление номеров:**
1. Точное совпадение
2. OCR-варианты (A↔Z, H↔N и др.)
3. Расстояние Левенштейна ≤ 1

**Вывод:**
- `cross_camera_speeds.json` — JSON с результатами
- `cross_camera_report.jpg` — визуальный отчёт (карточки + таблица)
- Консоль: таблица скоростей
