# Калибровка

Калибровка — ключевой шаг для точного измерения скорости. Для каждой камеры нужно настроить:
1. Гомографию (перспектива → метры)
2. `speed_correction` (поправочный коэффициент)
3. `cross_time_offset` (компенсация задержки камеры)
4. Расстояния между камерами (кросс-камера)
5. OCR-параметры (min_crop_width, plate_stretch_x)

## 1. Гомография

### Создание матрицы

```bash
# Захватить кадр с камеры
python tools/grab_frame.py --camera Camera_12 --output config/frame_Camera_12.jpg

# Запустить калибровку
python tools/calibrate_homography.py \
    --video config/frame_Camera_12.jpg \
    --camera Camera_12
```

**Процесс:**
1. Откроется окно с кадром
2. Кликните 4 точки, образующие прямоугольник на дороге (полотно, разметка)
3. Введите реальные размеры в метрах:
   - **Ширина** — расстояние между левыми и правыми точками (поперёк дороги)
   - **Длина** — расстояние между верхними и нижними точками (вдоль дороги)
4. Матрица гомографии сохраняется

**Советы:**
- Используйте разметку, бордюры, известные расстояния как ориентиры
- Чем дальше точки друг от друга — тем точнее матрица
- Все 4 точки должны лежать на одной плоскости (дорога)

**Выходные файлы:**
```
config/homography_Camera_12.yaml          # Конфигурация
config/homography_Camera_12_matrix.npy    # Матрица (NumPy)
```

### Подключение

В `config/config_cam.yaml` для камеры:

```yaml
cameras:
  - name: Camera_12
    homography_config: config/homography_Camera_12.yaml
```

## 2. speed_correction — Поправочный коэффициент

Гомография даёт приблизительные расстояния. `speed_correction` компенсирует систематическую ошибку.

### Замер через estimate_distance.py

```bash
python tools/estimate_distance.py \
    records/Camera_12/video.mp4 \
    --homography config/homography_Camera_12.yaml \
    --correction 1.0
```

**Процесс:**
1. Запустить с `--correction 1.0` (без коррекции)
2. В видео будут показаны расстояния, пройденные машинами (в метрах)
3. Сравнить с реальным расстоянием (измеренным на месте)
4. `speed_correction = реальное_расстояние / показанное_расстояние`

**Пример:**
- Реальное расстояние между двумя точками: 25 метров
- estimate_distance показывает: 31 метр
- `speed_correction = 25 / 31 = 0.806`

### Итеративная проверка

```bash
# Проверить с новым коэффициентом
python tools/estimate_distance.py video.mp4 \
    --homography config/homography_Camera_12.yaml \
    --correction 0.806
```

Теперь расстояния должны быть ближе к реальным. Повторять до приемлемой точности.

### Применение

В `config/config_cam.yaml`:

```yaml
cameras:
  - name: Camera_12
    speed_correction: 0.805
```

## 3. cross_time_offset — Компенсация задержки камеры

Камеры на WiFi могут иметь разную задержку передачи потока. Это влияет на кросс-камерный замер скорости.

### Как определить

1. Создать событие с известным временем (хлопок, жест) перед всеми камерами
2. Записать потоки всех камер одновременно: `python tools/record_stream.py --camera Camera_12 Camera_14 Camera_21`
3. Открыть записи, найти кадр события на каждой камере
4. Разница в timestamps = offset

**Пример:**
- Camera_12 зафиксировала событие на 0.0 с
- Camera_21 зафиксировала то же событие на +0.7 с
- `cross_time_offset: 0.7` для Camera_21

### Применение

```yaml
cameras:
  - name: Camera_21
    cross_time_offset: 0.7   # секунды, добавляется к timestamp
```

## 4. Расстояния между камерами

Для кросс-камерного замера нужны точные расстояния между камерами.

### Измерение

Измерить расстояние по дороге (не по прямой) между позициями, где камера фиксирует номер (зона OCR). Используйте:
- GPS-координаты + картографический сервис
- Рулетку/лазерный дальномер на месте
- Google Earth (линейка вдоль дороги)

### Конфигурация

В `config/config_cam.yaml`:

```yaml
cross_camera:
  enabled: true
  speed_limit: 70
  distances_m:
    - cameras: [Camera_12, Camera_14]
      distance: 41.3           # метры по дороге
    - cameras: [Camera_14, Camera_21]
      distance: 30.05
```

Непрямые расстояния суммируются автоматически:
- Camera_12 → Camera_21 = 41.3 + 30.05 = 71.35 м

## 5. OCR-параметры

### min_crop_width

Минимальная ширина кропа машины (пикселей) для запуска OCR. Слишком маленькие кропы — номер нечитаем.

**Подбор:**
1. Запустить обработку с `min_crop_width: 200`
2. Смотреть `debug_ocr/car_crops/` — при какой ширине OCR начинает ошибаться
3. Типичные значения: 300-400 px

```yaml
cameras:
  - name: Camera_12
    min_crop_width: 350
```

### min_plate_width

Минимальная ширина номерной пластины (пикселей). NomeroffNet ненадёжен при < 55 px.

**Рекомендации:**
- `>= 65 px` — надёжное распознавание
- `55-65 px` — допустимо с коррекцией
- `< 55 px` — слишком мало, OCR будет ошибаться

```yaml
cameras:
  - name: Camera_12
    min_plate_width: 30    # камера близко, номера крупные
  - name: Camera_21
    min_plate_width: 65    # камера далеко
```

### plate_stretch_x

Горизонтальное растяжение кропа номера перед повторным OCR (Smart Plate Re-OCR). Помогает когда камера под углом и номер сжат по горизонтали.

**Подбор:**
1. Смотреть `debug_ocr/plate_crops/` — номера выглядят сжатыми?
2. Попробовать 1.2 — 1.5
3. При `plate_stretch_x > 1.05` автоматически применяется лёгкий GaussianBlur

```yaml
cameras:
  - name: Camera_14
    plate_stretch_x: 1.3
  - name: Camera_21
    plate_stretch_x: 1.35
```

### crop_zone

Полигон, внутри которого OCR активен. Рисуется через `tools/draw_zones.py`:

```bash
python tools/draw_zones.py --camera Camera_12 --image config/frame_Camera_12.jpg
```

Ограничивает OCR областью дороги где номера читаемы (исключает далёкие участки, тротуары).

## Рекомендуемый порядок калибровки

1. `grab_frame.py` — захватить кадры со всех камер
2. `draw_zones.py` — настроить crop_zone для каждой камеры
3. `calibrate_homography.py` — создать матрицы гомографии
4. `estimate_distance.py` — подобрать `speed_correction`
5. `record_stream.py` — записать тестовый проезд
6. `cli.py --quality max` — обработать запись, проверить результаты
7. Итеративно подстроить `min_crop_width`, `plate_stretch_x`, `min_plate_width`
8. Измерить `cross_time_offset` и расстояния между камерами
