/**
 * CameraMarker Component - Camera location marker on map
 * Requirements: 4.2 (camera markers), 4.3 (click to open stream)
 */

import { useMemo } from 'react'
import { Marker, Popup } from 'react-leaflet'
import L from 'leaflet'
import type { Camera } from '../../types'
import { useTranslation } from 'react-i18next'

interface CameraMarkerProps {
  camera: Camera
  isSelected?: boolean
  isDraggable?: boolean
  onClick?: (camera: Camera) => void
  onDragEnd?: (camera: Camera, lat: number, lng: number) => void
}

// Create custom camera icon based on status
function createCameraIcon(
  status: Camera['status'],
  type: Camera['type'],
  isSelected: boolean
): L.DivIcon {
  const statusColors = {
    online: '#10b981', // success green
    offline: '#64748b', // neutral gray
    error: '#ef4444', // error red
  }

  const color = statusColors[status]
  const size = isSelected ? 40 : 32
  const borderWidth = isSelected ? 3 : 2
  const isSmart = type === 'smart'

  // SVG camera icon with status indicator
  const svgIcon = `
    <svg width="${size}" height="${size}" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Outer ring (status color) -->
      <circle cx="16" cy="16" r="14" fill="white" stroke="${color}" stroke-width="${borderWidth}"/>
      
      <!-- Camera body -->
      <rect x="8" y="11" width="12" height="10" rx="2" fill="${color}"/>
      
      <!-- Camera lens -->
      <circle cx="14" cy="16" r="3" fill="white"/>
      <circle cx="14" cy="16" r="1.5" fill="${color}"/>
      
      <!-- Camera flash/sensor (for smart cameras) -->
      ${isSmart ? `<rect x="21" y="13" width="3" height="6" rx="1" fill="${color}"/>` : ''}
      
      <!-- Status dot -->
      <circle cx="24" cy="8" r="4" fill="${color}" stroke="white" stroke-width="1.5"/>
    </svg>
  `

  return L.divIcon({
    html: svgIcon,
    className: `camera-marker ${isSelected ? 'camera-marker--selected' : ''}`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
  })
}

export function CameraMarker({
  camera,
  isSelected = false,
  isDraggable = false,
  onClick,
  onDragEnd,
}: CameraMarkerProps) {
  const { t } = useTranslation()

  const icon = useMemo(
    () => createCameraIcon(camera.status, camera.type, isSelected),
    [camera.status, camera.type, isSelected]
  )

  const handleClick = () => {
    if (onClick) {
      onClick(camera)
    }
  }

  const handleDragEnd = (e: L.DragEndEvent) => {
    if (onDragEnd) {
      const marker = e.target
      const position = marker.getLatLng()
      onDragEnd(camera, position.lat, position.lng)
    }
  }

  const statusText = {
    online: t('camera.online'),
    offline: t('camera.offline'),
    error: t('camera.error'),
  }

  const typeText = camera.type === 'smart' ? t('camera.smart') : t('camera.standard')

  return (
    <Marker
      position={[camera.location.lat, camera.location.lng]}
      icon={icon}
      draggable={isDraggable}
      eventHandlers={{
        click: handleClick,
        dragend: handleDragEnd,
      }}
    >
      <Popup className="camera-popup">
        <div className="min-w-[200px] p-1">
          {/* Camera name */}
          <h3 className="font-semibold text-neutral-900 text-sm mb-1">{camera.name}</h3>

          {/* Status badge */}
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                camera.status === 'online'
                  ? 'bg-success/10 text-success'
                  : camera.status === 'offline'
                    ? 'bg-neutral-100 text-neutral-600'
                    : 'bg-error/10 text-error'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                  camera.status === 'online'
                    ? 'bg-success'
                    : camera.status === 'offline'
                      ? 'bg-neutral-400'
                      : 'bg-error'
                }`}
              />
              {statusText[camera.status]}
            </span>
            <span className="text-xs text-neutral-500">{typeText}</span>
          </div>

          {/* Location */}
          {camera.location.address && (
            <p className="text-xs text-neutral-500 mb-2">{camera.location.address}</p>
          )}

          {/* View stream button */}
          <button
            onClick={handleClick}
            className="w-full px-3 py-1.5 bg-primary-500 hover:bg-primary-600 text-white text-xs font-medium rounded-lg transition-colors"
          >
            {t('map.viewStream', 'View Stream')}
          </button>
        </div>
      </Popup>
    </Marker>
  )
}
