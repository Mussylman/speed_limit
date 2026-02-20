/**
 * RouteVisualizer Component - Vehicle route visualization on map
 * Requirements: 5.3 (route polyline), 5.4 (lane info), 5.5 (start/end markers)
 */

import { useMemo } from 'react'
import { Polyline, Marker, Popup, CircleMarker } from 'react-leaflet'
import L from 'leaflet'
import type { VehicleRoute } from '../../types'
import { useTranslation } from 'react-i18next'

interface RouteVisualizerProps {
  route: VehicleRoute
  showTimestamps?: boolean
  animateRoute?: boolean
  color?: string
}

// Create start marker icon (green flag)
function createStartIcon(): L.DivIcon {
  const svg = `
    <svg width="32" height="40" viewBox="0 0 32 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M16 0C7.163 0 0 7.163 0 16c0 12 16 24 16 24s16-12 16-24c0-8.837-7.163-16-16-16z" fill="#10b981"/>
      <circle cx="16" cy="16" r="8" fill="white"/>
      <path d="M12 12v8l6-4-6-4z" fill="#10b981"/>
    </svg>
  `
  return L.divIcon({
    html: svg,
    className: 'route-marker route-marker--start',
    iconSize: [32, 40],
    iconAnchor: [16, 40],
    popupAnchor: [0, -40],
  })
}

// Create end marker icon (red flag)
function createEndIcon(): L.DivIcon {
  const svg = `
    <svg width="32" height="40" viewBox="0 0 32 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M16 0C7.163 0 0 7.163 0 16c0 12 16 24 16 24s16-12 16-24c0-8.837-7.163-16-16-16z" fill="#ef4444"/>
      <circle cx="16" cy="16" r="8" fill="white"/>
      <rect x="12" y="12" width="8" height="8" rx="1" fill="#ef4444"/>
    </svg>
  `
  return L.divIcon({
    html: svg,
    className: 'route-marker route-marker--end',
    iconSize: [32, 40],
    iconAnchor: [16, 40],
    popupAnchor: [0, -40],
  })
}

// Format timestamp for display
function formatTimestamp(date: Date): string {
  return new Intl.DateTimeFormat('default', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    day: '2-digit',
    month: 'short',
  }).format(date)
}

export function RouteVisualizer({
  route,
  showTimestamps = true,
  animateRoute = false,
  color = '#0ea5e9',
}: RouteVisualizerProps) {
  const { t } = useTranslation()

  // Extract route path coordinates
  const pathPositions = useMemo(() => {
    if (route.routePath.length > 0) {
      return route.routePath.map((loc) => [loc.lat, loc.lng] as [number, number])
    }
    // Fallback to detection locations if routePath is empty
    return route.detections.map(() => {
      // We need camera locations for this - for now use a placeholder
      return [0, 0] as [number, number]
    })
  }, [route.routePath, route.detections])

  // Get start and end points
  const startPoint = useMemo(() => {
    if (route.routePath.length > 0) return route.routePath[0]
    return null
  }, [route.routePath])

  const endPoint = useMemo(() => {
    if (route.routePath.length > 1) return route.routePath[route.routePath.length - 1]
    return null
  }, [route.routePath])

  // Get first and last detections for popup info
  const firstDetection = route.detections[0]
  const lastDetection = route.detections[route.detections.length - 1]

  const startIcon = useMemo(() => createStartIcon(), [])
  const endIcon = useMemo(() => createEndIcon(), [])

  if (pathPositions.length < 2) {
    return null
  }

  return (
    <>
      {/* Route polyline */}
      <Polyline
        positions={pathPositions}
        pathOptions={{
          color: color,
          weight: 4,
          opacity: 0.8,
          lineCap: 'round',
          lineJoin: 'round',
          dashArray: animateRoute ? '10, 10' : undefined,
          className: animateRoute ? 'route-line--animated' : '',
        }}
      />

      {/* Route shadow for depth */}
      <Polyline
        positions={pathPositions}
        pathOptions={{
          color: '#000',
          weight: 6,
          opacity: 0.15,
          lineCap: 'round',
          lineJoin: 'round',
        }}
      />

      {/* Start marker */}
      {startPoint && (
        <Marker position={[startPoint.lat, startPoint.lng]} icon={startIcon}>
          <Popup>
            <div className="min-w-[180px] p-1">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full bg-success" />
                <span className="font-semibold text-sm text-neutral-900">
                  {t('map.routeStart')}
                </span>
              </div>
              {firstDetection && (
                <>
                  <p className="text-xs text-neutral-600 mb-1">
                    {t('map.timestamp')}: {formatTimestamp(new Date(firstDetection.timestamp))}
                  </p>
                  <p className="text-xs text-neutral-600">
                    {t('map.lane')}: {firstDetection.lane}
                  </p>
                </>
              )}
            </div>
          </Popup>
        </Marker>
      )}

      {/* End marker */}
      {endPoint && (
        <Marker position={[endPoint.lat, endPoint.lng]} icon={endIcon}>
          <Popup>
            <div className="min-w-[180px] p-1">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full bg-error" />
                <span className="font-semibold text-sm text-neutral-900">{t('map.routeEnd')}</span>
              </div>
              {lastDetection && (
                <>
                  <p className="text-xs text-neutral-600 mb-1">
                    {t('map.timestamp')}: {formatTimestamp(new Date(lastDetection.timestamp))}
                  </p>
                  <p className="text-xs text-neutral-600">
                    {t('map.lane')}: {lastDetection.lane}
                  </p>
                </>
              )}
            </div>
          </Popup>
        </Marker>
      )}

      {/* Intermediate waypoints with timestamps */}
      {showTimestamps &&
        route.routePath.slice(1, -1).map((point, index) => {
          const detection = route.detections[index + 1]
          return (
            <CircleMarker
              key={`waypoint-${index}`}
              center={[point.lat, point.lng]}
              radius={6}
              pathOptions={{
                color: color,
                fillColor: 'white',
                fillOpacity: 1,
                weight: 2,
              }}
            >
              {detection && (
                <Popup>
                  <div className="min-w-[150px] p-1">
                    <p className="text-xs text-neutral-600 mb-1">
                      {t('map.timestamp')}: {formatTimestamp(new Date(detection.timestamp))}
                    </p>
                    <p className="text-xs text-neutral-600">
                      {t('map.lane')}: {detection.lane}
                    </p>
                  </div>
                </Popup>
              )}
            </CircleMarker>
          )
        })}
    </>
  )
}
