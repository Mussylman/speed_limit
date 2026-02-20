# Установка

## Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU | NVIDIA RTX 3050 (4 GB) | RTX 3060+ (8 GB) |
| RAM | 8 GB | 16 GB |
| ОС | Windows 10, Ubuntu 20.04 | Windows 11, Ubuntu 22.04 |
| Python | 3.10 | 3.10 |
| CUDA | 11.8 | 12.x |
| cuDNN | 8.6+ | 8.9+ |

## Зависимости

### Python-пакеты

```bash
pip install ultralytics          # YOLO11 + ByteTrack
pip install torch torchvision    # PyTorch с CUDA
pip install opencv-python        # OpenCV (или opencv-python-headless)
pip install pyyaml               # Конфигурация
pip install pillow               # Работа с изображениями
pip install numpy                # Массивы
```

### NomeroffNet

NomeroffNet не устанавливается через pip — клонируется локально и добавляется в `sys.path` через `src/config.py`.

```bash
# Клонировать в корень проекта
git clone https://github.com/ria-com/nomeroff-net.git nomeroff-net

# Установить зависимости NomeroffNet
pip install -r nomeroff-net/requirements.txt
```

**TensorRT engine** (для максимальной производительности):

NomeroffNet использует YOLO-детектор номерных пластин. Для RTX 3050 нужен TensorRT engine:

```
nomeroff-net/data/models/Detector/yolov11x/yolov11x-keypoints-2024-10-11.engine
```

Engine генерируется автоматически при первом запуске из `.pt` модели, либо конвертируется вручную через `trtexec`. Engine привязан к конкретной GPU и версии TensorRT.

### FFmpeg

FFmpeg используется для записи видео (GPU-кодирование через NVENC, fallback на CPU libx264).

**Windows:**
```bash
# Скачать с https://ffmpeg.org/download.html
# Добавить в PATH
ffmpeg -version
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Проверка GPU

```bash
# NVIDIA драйвер
nvidia-smi

# PyTorch видит GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Ожидаемый вывод:
# True NVIDIA GeForce RTX 3050 Laptop GPU
```

## Структура моделей

```
speed_limit/
├── models/
│   └── yolo11n.pt                # YOLO11 nano (скачивается автоматически)
└── nomeroff-net/
    └── data/models/Detector/
        └── yolov11x/
            └── yolov11x-keypoints-2024-10-11.engine  # TensorRT engine
```

## YOLO модели

По умолчанию используется `yolo11n.pt` (nano — быстрая). Для лучшего качества можно использовать `yolo11s.pt` или `yolo11m.pt`. Модель скачивается автоматически при первом запуске.

Путь к модели задаётся в `config/config.yaml`:
```yaml
models:
  yolo_model: "models/yolo11n.pt"
```

## Проверка установки

```bash
# Должен запуститься без ошибок и показать GPU info
python src/cli.py --source image --path config/frame_Camera_12.jpg --camera Camera_12
```
