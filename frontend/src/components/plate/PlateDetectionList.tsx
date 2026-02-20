/**
 * PlateDetectionList Component - Real-time plate detection feed
 * Requirements: 2.4 (plate detection list), 3.1 (click to vehicle detail)
 *
 * Displays recently recognized plates with timestamps and camera info
 * Clicking a plate navigates to vehicle detail page
 */

import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Clock, Camera, ChevronRight } from 'lucide-react'
import { PlateDisplay } from './PlateDisplay'
import { useCameraStore } from '../../stores/cameraStore'
import type { PlateDetection } from '../../types'
import { cn } from '../../utils/cn'

export interface PlateDetectionListProps {
  detections: PlateDetection[]
  maxItems?: number
  showCamera?: boolean
  className?: string
  emptyMessage?: string
}

/**
 * Formats timestamp for display
 */
function formatTime(date: Date): string {
  return new Intl.DateTimeFormat('default', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date)
}

/**
 * Formats date for display
 */
function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('default', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  }).format(date)
}

export function PlateDetectionList({
  detections,
  maxItems = 10,
  showCamera = true,
  className,
  emptyMessage,
}: PlateDetectionListProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const cameras = useCameraStore((state) => state.cameras)

  const displayedDetections = detections.slice(0, maxItems)

  const getCameraName = (cameraId: string): string => {
    const camera = cameras.find((c) => c.id === cameraId)
    return camera?.name || cameraId
  }

  const handlePlateClick = (plate: string) => {
    navigate(`/vehicles/${encodeURIComponent(plate)}`)
  }

  if (displayedDetections.length === 0) {
    return (
      <div className={cn('flex flex-col items-center justify-center py-12 text-center', className)}>
        <div className="w-16 h-16 rounded-full bg-neutral-100 flex items-center justify-center mb-4">
          <Camera className="w-8 h-8 text-neutral-400" />
        </div>
        <p className="text-neutral-500 body-base">
          {emptyMessage || t('plate.noDetections', 'No plate detections yet')}
        </p>
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      {displayedDetections.map((detection, index) => (
        <div
          key={detection.id}
          className={cn(
            'group flex items-center gap-4 p-3 rounded-xl',
            'bg-white border border-neutral-200',
            'hover:border-primary-300 hover:shadow-md',
            'transition-all duration-200 cursor-pointer',
            'animate-fadeIn'
          )}
          style={{ animationDelay: `${index * 50}ms` }}
          onClick={() => handlePlateClick(detection.plate)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              handlePlateClick(detection.plate)
            }
          }}
        >
          {/* Plate Display */}
          <PlateDisplay plate={detection.plate} size="small" />

          {/* Detection Info */}
          <div className="flex-1 min-w-0">
            {/* Time */}
            <div className="flex items-center gap-1.5 text-neutral-700">
              <Clock className="w-3.5 h-3.5 text-neutral-400" />
              <span className="text-sm font-medium">{formatTime(detection.timestamp)}</span>
              <span className="text-xs text-neutral-400">{formatDate(detection.timestamp)}</span>
            </div>

            {/* Camera */}
            {showCamera && (
              <div className="flex items-center gap-1.5 mt-1 text-neutral-500">
                <Camera className="w-3.5 h-3.5 text-neutral-400" />
                <span className="text-xs truncate">{getCameraName(detection.cameraId)}</span>
                {detection.lane && (
                  <span className="text-xs text-neutral-400">
                    • {t('plate.lane', 'Lane')} {detection.lane}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Confidence Badge */}
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-xs font-medium px-2 py-0.5 rounded-full',
                detection.confidence >= 0.9
                  ? 'bg-success-100 text-success-700'
                  : detection.confidence >= 0.7
                    ? 'bg-warning-100 text-warning-700'
                    : 'bg-neutral-100 text-neutral-600'
              )}
            >
              {Math.round(detection.confidence * 100)}%
            </span>

            {/* Arrow */}
            <ChevronRight className="w-4 h-4 text-neutral-300 group-hover:text-primary-500 transition-colors" />
          </div>
        </div>
      ))}
    </div>
  )
}
