# 🔌 SERGEK Frontend - Руководство по интеграции с Backend

## 📋 Общий обзор

Данная документация подробно описывает, как интегрировать frontend приложение SERGEK с backend системой. Frontend полностью готов к работе с backend, mock данные удалены, API сервисы подготовлены.

## 🏗️ Текущая архитектура

### Frontend технологии
- **React 19.2.0** + TypeScript 5.9.3
- **Vite 7.2.5** (Инструмент сборки)
- **TailwindCSS 4.1.18** (Стилизация)
- **React Router DOM 7.12.0** (Навигация)
- **Zustand 5.0.10** (Управление состоянием)
- **i18next 25.7.4** (Интернационализация)
- **HLS.js 1.6.15** (Видео стриминг)
- **Leaflet 1.9.4** (Интерактивные карты)
- **Framer Motion 12.26.2** (Анимации)

### API сервисы
В frontend реализованы 3 основных API сервиса:
- `cameraService` - Управление камерами
- `violationService` - Управление нарушениями  
- `vehicleService` - Транспорт и распознавание номеров

## 🔌 API Endpoints

### 1. Сервис камер (`/api/cameras`)

#### GET `/api/cameras`
Получить список всех камер
```json
Response: [
  {
    "id": "string",
    "name": "string", 
    "rtsp_url": "string",
    "hls_url": "string",
    "location": {
      "lat": number,
      "lng": number,
      "address": "string"
    },
    "type": "smart" | "standard",
    "status": "online" | "offline" | "error",
    "lane": number
  }
]
```

#### GET `/api/cameras/{id}`
Получить конкретную камеру по ID

#### POST `/api/cameras`
Создать новую камеру
```json
Request: {
  "name": "string",
  "rtsp_url": "string", 
  "type": "smart" | "standard",
  "location": {
    "lat": number,
    "lng": number,
    "address": "string"
  },
  "lane": number
}
```

#### PATCH `/api/cameras/{id}`
Обновить информацию о камере

#### DELETE `/api/cameras/{id}`
Удалить камеру

#### PATCH `/api/cameras/{id}/status`
Обновить статус камеры
```json
Request: {
  "status": "online" | "offline" | "error"
}
```

#### GET `/api/cameras/{id}/stream`
Получить URL потока камеры
```json
Response: {
  "hls_url": "string"
}
```

#### POST `/api/cameras/test-connection`
Тестировать RTSP подключение
```json
Request: {
  "rtsp_url": "string"
}
Response: {
  "success": boolean,
  "message": "string"
}
```

#### GET `/api/cameras?type=smart`
Получить только умные камеры

### 2. Сервис нарушений (`/api/violations`)

#### GET `/api/violations`
Получить список нарушений (с поддержкой фильтрации)
Query Parameters:
- `date_from`: ISO date string
- `date_to`: ISO date string  
- `type`: тип нарушения
- `plate`: номер автомобиля
- `status`: статус нарушения
- `limit`: количество записей
- `offset`: смещение для пагинации

```json
Response: [
  {
    "id": "string",
    "type": "speed_limit" | "red_light" | "wrong_lane" | "no_seatbelt" | "phone_usage" | "parking" | "other",
    "plate": "string",
    "camera_id": "string", 
    "timestamp": "ISO date string",
    "image_url": "string",
    "video_clip_url": "string",
    "status": "pending" | "confirmed" | "dismissed",
    "fine": number,
    "description": "string",
    "location": {
      "lat": number,
      "lng": number,
      "address": "string"
    }
  }
]
```

#### GET `/api/violations/{id}`
Получить конкретное нарушение

#### PATCH `/api/violations/{id}/status`
Обновить статус нарушения
```json
Request: {
  "status": "pending" | "confirmed" | "dismissed"
}
```

#### GET `/api/violations/stats`
Получить статистику нарушений
```json
Response: {
  "total": number,
  "pending": number,
  "confirmed": number,
  "dismissed": number,
  "today": number,
  "byType": {
    "speed_limit": number,
    "red_light": number,
    "wrong_lane": number,
    "no_seatbelt": number,
    "phone_usage": number,
    "parking": number,
    "other": number
  }
}
```

### 3. Сервис транспорта (`/api/vehicles`)

#### GET `/api/vehicles/{plate}`
Получить информацию о транспорте по номеру
```json
Response: {
  "id": "string",
  "plate": "string",
  "brand": "string",
  "model": "string", 
  "color": "string",
  "year": number,
  "owner": {
    "name": "string",
    "iin": "string",
    "phone": "string",
    "address": "string"
  }
}
```

#### GET `/api/vehicles/search?q={query}`
Поиск транспорта по частичному номеру

#### GET `/api/vehicles/{plate}/route`
Получить маршрут движения транспорта
```json
Response: {
  "plate": "string",
  "detections": [
    {
      "camera_id": "string",
      "camera_name": "string",
      "location": {
        "lat": number,
        "lng": number,
        "address": "string"
      },
      "timestamp": "ISO date string",
      "lane": number
    }
  ],
  "total_distance": number,
  "duration": number
}
```

#### GET `/api/detections`
Получить записи распознавания номеров
Query Parameters:
- `limit`: number (по умолчанию 50)
- `plate`: string
- `camera_id`: string
- `date_from`: ISO date string
- `date_to`: ISO date string

```json
Response: [
  {
    "id": "string",
    "plate": "string",
    "camera_id": "string",
    "camera_name": "string",
    "timestamp": "ISO date string",
    "confidence": number,
    "image_url": "string", 
    "lane": number,
    "location": {
      "lat": number,
      "lng": number,
      "address": "string"
    }
  }
]
```

#### GET `/api/detections/recent?limit={number}`
Получить последние распознавания номеров

#### GET `/api/detections/stats`
Получить статистику распознавания
```json
Response: {
  "total_today": number,
  "total_week": number,
  "total_month": number,
  "accuracy_rate": number,
  "by_camera": [
    {
      "camera_id": "string",
      "camera_name": "string", 
      "count": number
    }
  ]
}
```

## 🎥 Видео стриминг

### Конвертация RTSP в HLS
Backend должен обеспечивать следующие возможности:

1. **Прием RTSP потоков**: Получение RTSP потоков от камер
2. **Конвертация в HLS**: Преобразование RTSP в HLS формат (можно использовать FFmpeg)
3. **Stream Endpoint**: Endpoint `/api/cameras/{id}/stream` возвращает HLS URL

### Пример формата HLS URL
```
https://your-backend.com/streams/{camera-id}/playlist.m3u8
```

### Frontend Video Player
- Использует библиотеку **HLS.js**
- Автоматическая настройка качества
- Поддержка полноэкранного режима
- Управление звуком
- Индикатор состояния подключения
- Обработка ошибок подключения

### Рекомендуемая конфигурация FFmpeg
```bash
ffmpeg -i {rtsp_url} \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -c:a aac -ar 44100 -ac 2 \
  -f hls -hls_time 2 -hls_list_size 3 -hls_flags delete_segments \
  /path/to/streams/{camera_id}/playlist.m3u8
```

## 🔧 Конфигурация

### API Base URL
В файле `src/services/api/client.ts` настраивается базовый URL API:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})
```

### Переменные окружения
Необходимо определить в файле `.env`:
```env
# API Configuration
VITE_API_BASE_URL=https://your-backend-api.com/api
VITE_WS_URL=wss://your-backend-api.com/ws

# Map Configuration (Шымкент)
VITE_MAP_CENTER_LAT=42.3417
VITE_MAP_CENTER_LNG=69.5901
VITE_MAP_DEFAULT_ZOOM=13

# Video Streaming
VITE_HLS_TIMEOUT=30000
VITE_RTSP_TIMEOUT=10000

# Development
VITE_DEV_MODE=false
VITE_LOG_LEVEL=error
```

## 📡 Функции реального времени

### WebSocket подключения
Для обновлений в реальном времени поддержка WebSocket:

1. **Новые нарушения**: Новые нарушения от умных камер
2. **Статус камер**: Статус камер online/offline
3. **Распознавание номеров**: Результаты распознавания в реальном времени
4. **Системные уведомления**: Важные системные события

### Рекомендуемые WebSocket события
```typescript
// От frontend к backend
{
  "type": "subscribe_camera_status",
  "camera_id": "string"
}

{
  "type": "subscribe_violations",
  "filters": {
    "status": "pending",
    "type": "speed_limit"
  }
}

// От backend к frontend  
{
  "type": "camera_status_update",
  "camera_id": "string", 
  "status": "online" | "offline" | "error",
  "timestamp": "ISO date string"
}

{
  "type": "new_violation",
  "violation": ViolationObject
}

{
  "type": "new_detection", 
  "detection": DetectionObject
}

{
  "type": "system_notification",
  "level": "info" | "warning" | "error",
  "message": "string",
  "timestamp": "ISO date string"
}
```

## �️ Рекомендации по схеме базы данных

### Таблица cameras
```sql
CREATE TABLE cameras (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  rtsp_url VARCHAR(500) NOT NULL,
  hls_url VARCHAR(500),
  latitude DECIMAL(10, 8) NOT NULL,
  longitude DECIMAL(11, 8) NOT NULL, 
  address TEXT,
  type ENUM('smart', 'standard') NOT NULL DEFAULT 'standard',
  status ENUM('online', 'offline', 'error') DEFAULT 'offline',
  lane INTEGER DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_cameras_type (type),
  INDEX idx_cameras_status (status),
  INDEX idx_cameras_location (latitude, longitude)
);
```

### Таблица violations
```sql
CREATE TABLE violations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type ENUM('speed_limit', 'red_light', 'wrong_lane', 'no_seatbelt', 'phone_usage', 'parking', 'other') NOT NULL,
  plate VARCHAR(20) NOT NULL,
  camera_id UUID NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  image_url VARCHAR(500) NOT NULL,
  video_clip_url VARCHAR(500),
  status ENUM('pending', 'confirmed', 'dismissed') DEFAULT 'pending',
  fine DECIMAL(10, 2),
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE,
  INDEX idx_violations_plate (plate),
  INDEX idx_violations_timestamp (timestamp),
  INDEX idx_violations_status (status),
  INDEX idx_violations_type (type),
  INDEX idx_violations_camera (camera_id)
);
```

### Таблица vehicles
```sql
CREATE TABLE vehicles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  plate VARCHAR(20) UNIQUE NOT NULL,
  brand VARCHAR(100),
  model VARCHAR(100),
  color VARCHAR(50),
  year INTEGER,
  owner_name VARCHAR(255),
  owner_iin VARCHAR(12),
  owner_phone VARCHAR(20),
  owner_address TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_vehicles_plate (plate),
  INDEX idx_vehicles_owner_iin (owner_iin)
);
```

### Таблица detections
```sql
CREATE TABLE detections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  plate VARCHAR(20) NOT NULL,
  camera_id UUID NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  image_url VARCHAR(500) NOT NULL,
  lane INTEGER DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE,
  INDEX idx_detections_plate (plate),
  INDEX idx_detections_timestamp (timestamp),
  INDEX idx_detections_camera (camera_id),
  INDEX idx_detections_confidence (confidence)
);
```

### Таблица system_logs (рекомендуется)
```sql
CREATE TABLE system_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  level ENUM('info', 'warning', 'error', 'debug') NOT NULL,
  message TEXT NOT NULL,
  component VARCHAR(100),
  user_id UUID,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_logs_level (level),
  INDEX idx_logs_timestamp (created_at),
  INDEX idx_logs_component (component)
);
```

## 🚀 Руководство по развертыванию

### Сборка Frontend
```bash
# Установка зависимостей
npm install

# Сборка для продакшена
npm run build

# Файлы сборки создаются в папке dist/
```

### Конфигурация Nginx
```nginx
server {
    listen 80;
    server_name sergek.shymkent.kz;
    
    # Перенаправление на HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name sergek.shymkent.kz;
    
    # SSL сертификаты
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;
    
    # Frontend статические файлы
    root /var/www/sergek-frontend/dist;
    index index.html;
    
    # Gzip сжатие
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # SPA маршрутизация
    location / {
        try_files $uri $uri/ /index.html;
        
        # Кэширование статических файлов
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # API прокси
    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Таймауты
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # WebSocket прокси
    location /ws/ {
        proxy_pass http://backend:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # HLS потоки
    location /streams/ {
        proxy_pass http://streaming-server:8080/streams/;
        
        # CORS для видео
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
        add_header Access-Control-Allow-Headers 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range';
        
        # Кэширование HLS сегментов
        location ~* \.(m3u8|ts)$ {
            expires 10s;
            add_header Cache-Control "no-cache";
        }
    }
}
```

### Docker Compose пример
```yaml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - backend
    networks:
      - sergek-network

  backend:
    image: sergek-backend:latest
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/sergek
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - sergek-network

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=sergek
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - sergek-network

  redis:
    image: redis:7-alpine
    networks:
      - sergek-network

volumes:
  postgres_data:

networks:
  sergek-network:
    driver: bridge
```

## 🔒 Рекомендации по безопасности

### Аутентификация и авторизация
Frontend готов к работе с JWT токенами:
```typescript
// Автоматическое добавление токена в API запросы
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Обработка истечения токена
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)
```

### CORS настройки
Backend должен настроить CORS:
```javascript
// Express.js пример
app.use(cors({
  origin: [
    'http://localhost:5173', // Development
    'https://sergek.shymkent.kz' // Production
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}))
```

### Валидация данных
```typescript
// Пример валидации на backend (Node.js + Joi)
const cameraSchema = Joi.object({
  name: Joi.string().min(3).max(255).required(),
  rtsp_url: Joi.string().uri().pattern(/^rtsp:\/\//).required(),
  type: Joi.string().valid('smart', 'standard').required(),
  location: Joi.object({
    lat: Joi.number().min(-90).max(90).required(),
    lng: Joi.number().min(-180).max(180).required(),
    address: Joi.string().max(500).optional()
  }).required(),
  lane: Joi.number().integer().min(1).max(10).optional()
})
```

## 📊 Оптимизация производительности

### Lazy Loading компонентов
Страницы загружаются по требованию:
```typescript
const CamerasPage = lazy(() => import('./pages/CamerasPage'))
const AdminPage = lazy(() => import('./pages/AdminPage'))
```

### Оптимизация изображений
Для изображений с камер:
- Использование WebP формата
- Создание thumbnail'ов для предварительного просмотра
- Использование CDN для статических файлов
- Lazy loading изображений

### Кэширование
- **API ответы**: Redis кэш на backend
- **Статические файлы**: Browser cache + CDN
- **HLS сегменты**: CDN кэш с коротким TTL
- **Изображения**: Long-term browser cache

### Оптимизация базы данных
```sql
-- Индексы для частых запросов
CREATE INDEX CONCURRENTLY idx_violations_recent 
ON violations (timestamp DESC, status) 
WHERE timestamp > NOW() - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY idx_detections_recent 
ON detections (timestamp DESC, camera_id) 
WHERE timestamp > NOW() - INTERVAL '7 days';

-- Партиционирование больших таблиц
CREATE TABLE detections_2024_01 PARTITION OF detections
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

## 🧪 Тестовые сценарии

### API тестирование
```bash
# Получить список камер
curl -X GET "http://localhost:8000/api/cameras" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Создать новую камеру
curl -X POST "http://localhost:8000/api/cameras" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "Тестовая камера",
    "rtsp_url": "rtsp://example.com/stream",
    "type": "smart",
    "location": {
      "lat": 42.3417, 
      "lng": 69.5901, 
      "address": "пр. Абая 45, Шымкент"
    },
    "lane": 1
  }'

# Получить нарушения с фильтрацией
curl -X GET "http://localhost:8000/api/violations?status=pending&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Тест RTSP подключения
curl -X POST "http://localhost:8000/api/cameras/test-connection" \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url": "rtsp://example.com/stream"}'
```

### Нагрузочное тестирование
```bash
# Apache Bench пример
ab -n 1000 -c 10 -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/api/cameras

# Тестирование WebSocket
wscat -c ws://localhost:8000/ws
```

## 🌐 Многоязычность (i18n)

### Поддерживаемые языки
- **Русский (ru)** - По умолчанию
- **Казахский (kk)** - Государственный язык
- **Английский (en)** - Международный

### API локализация
Backend должен поддерживать заголовок `Accept-Language`:
```typescript
// Frontend автоматически отправляет
headers: {
  'Accept-Language': 'ru,kk;q=0.9,en;q=0.8'
}
```

### Локализованные ответы
```json
// Пример локализованного ответа об ошибке
{
  "error": {
    "code": "CAMERA_NOT_FOUND",
    "message": {
      "ru": "Камера не найдена",
      "kk": "Камера табылмады", 
      "en": "Camera not found"
    }
  }
}
```

## 📱 Мобильная адаптация

### Responsive дизайн
Frontend полностью адаптивен:
- Мобильные устройства (320px+)
- Планшеты (768px+)
- Десктоп (1024px+)
- Большие экраны (1440px+)

### PWA поддержка (рекомендуется)
```json
// manifest.json
{
  "name": "SERGEK - Система видеонаблюдения",
  "short_name": "SERGEK",
  "description": "Система видеонаблюдения города Шымкент",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#3b82f6",
  "icons": [
    {
      "src": "/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

## 🔍 Мониторинг и логирование

### Frontend логирование
```typescript
// Отправка ошибок на backend
window.addEventListener('error', (event) => {
  fetch('/api/logs/frontend-error', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: event.error.message,
      stack: event.error.stack,
      url: window.location.href,
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString()
    })
  })
})
```

### Метрики производительности
```typescript
// Web Vitals отправка
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals'

getCLS(sendToAnalytics)
getFID(sendToAnalytics)
getFCP(sendToAnalytics)
getLCP(sendToAnalytics)
getTTFB(sendToAnalytics)
```

## 📞 Поддержка и контакты

Данная документация содержит всю необходимую информацию для интеграции frontend с backend системой. 

### Статус готовности
- ✅ **Frontend код**: Полностью готов и production-ready
- ✅ **API endpoints**: Детально описаны с примерами
- ✅ **Схема БД**: Рекомендации предоставлены
- ✅ **Deployment**: Конфигурации Nginx готовы
- ✅ **Безопасность**: JWT интеграция реализована
- ✅ **Многоязычность**: 3 языка поддерживаются
- ✅ **Видео стриминг**: HLS.js интегрирован
- ✅ **Карты**: Leaflet настроен для Шымкента

### Дополнительная помощь
- **Техническая поддержка**: support@sergek.kz
- **Документация API**: https://api.sergek.kz/docs
- **GitHub Issues**: Создать issue в репозитории

---

**Frontend приложение полностью готово к интеграции с backend системой!**

*Разработано для системы безопасности дорожного движения города Шымкент* 🚦

