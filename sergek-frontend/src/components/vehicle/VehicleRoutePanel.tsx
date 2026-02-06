/**
 * VehicleRoutePanel Component - Camera Transit List with Route Visualization
 * Requirements: 5.1, 5.2, 5.5 - Display camera passes with timestamps, from/to info
 *
 * Aesthetic: Arctic Command Center - Timeline-based route visualization
 */

import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { MapPin, Clock, Camera, Route, ArrowRight, Navigation, ChevronRight } from 'lucide-react'
import { PlateDisplay } from '../plate'
import { useCameraStore } from '../../stores/cameraStore'
import type { VehicleRoute, PlateDetection, Camera as CameraType } from '../../types'

export interface VehicleRoutePanelProps {
  route: VehicleRoute
  onShowOnMap?: () => void
  className?: string
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('ru-RU', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatDate(date: Date): string {
  return date.toLocaleDateString('ru-RU', {
    day: '2-digit',
    month: 'short',
  })
}

function formatDuration(startDate: Date, endDate: Date): string {
  const diffMs = endDate.getTime() - startDate.getTime()
  const diffMins = Math.floor(diffMs / (1000 * 60))
  const hours = Math.floor(diffMins / 60)
  const mins = diffMins % 60

  if (hours > 0) {
    return `${hours}ч ${mins}мин`
  }
  return `${mins}мин`
}

export function VehicleRoutePanel({ route, onShowOnMap, className = '' }: VehicleRoutePanelProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const getCameraById = useCameraStore((state) => state.getCameraById)

  // Sort detections chronologically (oldest first for route display)
  const sortedDetections = useMemo(
    () => [...route.detections].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()),
    [route.detections]
  )

  const firstDetection = sortedDetections[0]
  const lastDetection = sortedDetections[sortedDetections.length - 1]

  const firstCamera = firstDetection ? getCameraById(firstDetection.cameraId) : undefined
  const lastCamera = lastDetection ? getCameraById(lastDetection.cameraId) : undefined

  const duration =
    firstDetection && lastDetection
      ? formatDuration(firstDetection.timestamp, lastDetection.timestamp)
      : null

  if (sortedDetections.length === 0) {
    return (
      <div className={`card p-6 ${className}`}>
        <div className="text-center py-8">
          <Route className="w-12 h-12 text-neutral-300 mx-auto mb-3" />
          <p className="body-sm text-neutral-500">{t('common.noData')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`card overflow-hidden ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-neutral-200 bg-neutral-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Route className="w-5 h-5 text-primary-500" />
            <h3 className="heading-4">{t('vehicle.routePanel')}</h3>
          </div>
          <PlateDisplay plate={route.plate} size="small" />
        </div>
      </div>

      {/* From/To Summary */}
      <div className="p-4 bg-gradient-to-r from-primary-50 to-transparent border-b border-neutral-200">
        <div className="flex items-center gap-4">
          {/* From */}
          <div className="flex-1">
            <p className="caption text-neutral-500 mb-1">{t('vehicle.from')}</p>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-success-500 ring-2 ring-success-200" />
              <span className="body-sm font-medium text-neutral-800 truncate">
                {firstCamera?.name || `Camera ${firstDetection?.cameraId}`}
              </span>
            </div>
            {firstDetection && (
              <p className="caption text-neutral-400 mt-1 ml-5">
                {formatTime(firstDetection.timestamp)} • {formatDate(firstDetection.timestamp)}
              </p>
            )}
          </div>

          {/* Arrow */}
          <div className="flex flex-col items-center px-2">
            <ArrowRight className="w-5 h-5 text-neutral-400" />
            {duration && <span className="caption text-neutral-500 mt-1">{duration}</span>}
          </div>

          {/* To */}
          <div className="flex-1 text-right">
            <p className="caption text-neutral-500 mb-1">{t('vehicle.to')}</p>
            <div className="flex items-center justify-end gap-2">
              <span className="body-sm font-medium text-neutral-800 truncate">
                {lastCamera?.name || `Camera ${lastDetection?.cameraId}`}
              </span>
              <div className="w-3 h-3 rounded-full bg-error-500 ring-2 ring-error-200" />
            </div>
            {lastDetection && (
              <p className="caption text-neutral-400 mt-1 mr-5">
                {formatTime(lastDetection.timestamp)} • {formatDate(lastDetection.timestamp)}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Camera passes timeline */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="body-sm font-medium text-neutral-700 flex items-center gap-2">
            <Camera className="w-4 h-4" />
            {t('vehicle.passedCameras')}
          </h4>
          <span className="badge badge--info">{sortedDetections.length}</span>
        </div>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-[11px] top-3 bottom-3 w-0.5 bg-neutral-200" />

          {/* Timeline items */}
          <div className="space-y-1">
            {sortedDetections.map((detection, index) => (
              <RouteTimelineItem
                key={detection.id}
                detection={detection}
                camera={getCameraById(detection.cameraId)}
                isFirst={index === 0}
                isLast={index === sortedDetections.length - 1}
                onClick={() => navigate(`/cameras/${detection.cameraId}`)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Show on map button */}
      {onShowOnMap && (
        <div className="p-4 border-t border-neutral-200 bg-neutral-50">
          <button
            onClick={onShowOnMap}
            className="btn btn--primary w-full flex items-center justify-center gap-2"
          >
            <Navigation className="w-4 h-4" />
            {t('vehicle.showOnMap')}
          </button>
        </div>
      )}
    </div>
  )
}

// Timeline item component
interface RouteTimelineItemProps {
  detection: PlateDetection
  camera?: CameraType
  isFirst: boolean
  isLast: boolean
  onClick: () => void
}

function RouteTimelineItem({
  detection,
  camera,
  isFirst,
  isLast,
  onClick,
}: RouteTimelineItemProps) {
  const { t } = useTranslation()

  return (
    <div
      className="relative flex items-start gap-3 p-2 rounded-lg hover:bg-neutral-50 cursor-pointer transition-colors group"
      onClick={onClick}
    >
      {/* Timeline dot */}
      <div
        className={`
          relative z-10 w-6 h-6 rounded-full flex items-center justify-center
          ${isFirst ? 'bg-success-500 text-white' : ''}
          ${isLast && !isFirst ? 'bg-error-500 text-white' : ''}
          ${!isFirst && !isLast ? 'bg-white border-2 border-primary-400' : ''}
        `}
      >
        {isFirst && <MapPin className="w-3 h-3" />}
        {isLast && !isFirst && <MapPin className="w-3 h-3" />}
        {!isFirst && !isLast && <div className="w-2 h-2 rounded-full bg-primary-400" />}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 pt-0.5">
        <div className="flex items-center justify-between">
          <span className="body-sm font-medium text-neutral-800 truncate">
            {camera?.name || `Camera ${detection.cameraId}`}
          </span>
          <ChevronRight className="w-4 h-4 text-neutral-400 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
        <div className="flex items-center gap-3 mt-0.5">
          <span className="caption text-neutral-500 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {formatTime(detection.timestamp)}
          </span>
          <span className="caption text-neutral-400">
            {t('plate.lane')} {detection.lane}
          </span>
          <span className="caption text-neutral-400">{Math.round(detection.confidence)}%</span>
        </div>
      </div>
    </div>
  )
}
