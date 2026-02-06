/**
 * CameraCard Component
 * Displays a single camera feed with status indicator and overlay information
 * Requirements: 1.3, 1.5
 */

import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Video, VideoOff, AlertTriangle, MapPin } from 'lucide-react'
import { VideoPlayer } from './VideoPlayer'
import type { Camera } from '../../types'

interface CameraCardProps {
  camera: Camera
  isSelected?: boolean
  showOverlay?: boolean
  enableStream?: boolean
  onClick?: () => void
}

export function CameraCard({
  camera,
  isSelected = false,
  showOverlay = true,
  enableStream = false,
  onClick,
}: CameraCardProps) {
  const { t } = useTranslation()
  const [streamError, setStreamError] = useState(false)

  const statusConfig = {
    online: {
      color: 'bg-success-500',
      pulseColor: 'bg-success-500',
      label: t('camera.online'),
      icon: Video,
      ringColor: 'ring-success-500/20',
    },
    offline: {
      color: 'bg-neutral-400',
      pulseColor: 'bg-neutral-400',
      label: t('camera.offline'),
      icon: VideoOff,
      ringColor: 'ring-neutral-400/20',
    },
    error: {
      color: 'bg-error-500',
      pulseColor: 'bg-error-500',
      label: t('camera.error'),
      icon: AlertTriangle,
      ringColor: 'ring-error-500/20',
    },
  }

  const effectiveStatus = streamError ? 'error' : camera.status
  const status = statusConfig[effectiveStatus]
  const StatusIcon = status.icon

  const shouldShowStream =
    enableStream && camera.status === 'online' && camera.hlsUrl && !streamError

  return (
    <div
      onClick={onClick}
      className={`
        group relative overflow-hidden rounded-xl cursor-pointer
        transition-all duration-300 ease-out
        ${
          isSelected
            ? 'ring-2 ring-primary-500 ring-offset-2 ring-offset-neutral-50 shadow-lg scale-[1.02]'
            : 'hover:shadow-xl hover:scale-[1.01]'
        }
      `}
    >
      {/* Video / Stream Container */}
      <div className="relative aspect-video bg-neutral-900 overflow-hidden">
        {/* Gradient overlay for depth */}
        <div className="absolute inset-0 bg-gradient-to-t from-neutral-900/80 via-transparent to-neutral-900/20 z-10 pointer-events-none" />

        {/* Video stream when enabled and online */}
        {shouldShowStream ? (
          <VideoPlayer
            hlsUrl={camera.hlsUrl}
            autoPlay={true}
            muted={true}
            onError={() => setStreamError(true)}
          />
        ) : (
          <>
            {/* Placeholder pattern when offline/error */}
            {camera.status !== 'online' && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="absolute inset-0 opacity-10">
                  <div
                    className="w-full h-full"
                    style={{
                      backgroundImage: `repeating-linear-gradient(
                        45deg,
                        transparent,
                        transparent 10px,
                        rgba(255,255,255,0.03) 10px,
                        rgba(255,255,255,0.03) 20px
                      )`,
                    }}
                  />
                </div>
                <div className="relative z-20 flex flex-col items-center gap-3 text-neutral-400">
                  <StatusIcon className="w-12 h-12 opacity-50" />
                  <span className="text-sm font-medium">{status.label}</span>
                </div>
              </div>
            )}

            {/* Online camera - video placeholder (when stream not enabled) */}
            {camera.status === 'online' && (
              <div className="absolute inset-0 bg-neutral-800">
                {/* Simulated video feed placeholder */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative">
                    <Video className="w-16 h-16 text-neutral-600" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-3 h-3 bg-error-500 rounded-full animate-pulse" />
                    </div>
                  </div>
                </div>
                {/* Scan line effect */}
                <div
                  className="absolute inset-0 pointer-events-none opacity-30"
                  style={{
                    background:
                      'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.1) 2px, rgba(0,0,0,0.1) 4px)',
                  }}
                />
              </div>
            )}
          </>
        )}

        {/* Status indicator - top right */}
        <div className="absolute top-3 right-3 z-20">
          <div
            className={`
            flex items-center gap-2 px-2.5 py-1 rounded-full
            bg-neutral-900/70 backdrop-blur-sm
            border border-neutral-700/50
          `}
          >
            <div className="relative">
              <div className={`w-2 h-2 rounded-full ${status.color}`} />
              {effectiveStatus === 'online' && (
                <div
                  className={`absolute inset-0 w-2 h-2 rounded-full ${status.pulseColor} animate-ping`}
                />
              )}
            </div>
            <span className="text-xs font-medium text-neutral-200">{status.label}</span>
          </div>
        </div>

        {/* Smart camera badge */}
        {camera.type === 'smart' && (
          <div className="absolute top-3 left-3 z-20">
            <div className="px-2.5 py-1 rounded-full bg-primary-500/90 backdrop-blur-sm">
              <span className="text-xs font-semibold text-white tracking-wide">SMART</span>
            </div>
          </div>
        )}

        {/* Bottom overlay with camera info */}
        {showOverlay && (
          <div className="absolute bottom-0 left-0 right-0 z-20 p-4">
            <div className="flex items-end justify-between gap-4">
              <div className="flex-1 min-w-0">
                <h3 className="text-white font-semibold text-base truncate mb-1 font-display">
                  {camera.name}
                </h3>
                {camera.location.address && (
                  <div className="flex items-center gap-1.5 text-neutral-300">
                    <MapPin className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="text-xs truncate">{camera.location.address}</span>
                  </div>
                )}
              </div>

              {/* Camera ID badge */}
              <div className="flex-shrink-0">
                <span className="font-mono text-xs text-neutral-400 bg-neutral-800/80 px-2 py-1 rounded">
                  #{camera.id.slice(-4).toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Hover effect overlay */}
        <div
          className={`
          absolute inset-0 z-10 transition-opacity duration-300
          bg-primary-500/10 opacity-0 group-hover:opacity-100
          pointer-events-none
        `}
        />
      </div>
    </div>
  )
}
