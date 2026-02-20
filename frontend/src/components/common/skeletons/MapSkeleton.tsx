/**
 * Map Skeleton
 * Loading placeholder for map component
 * Requirements: 8.3 - Map loading state
 */

import { useMemo } from 'react'
import { Map, MapPin } from 'lucide-react'
import { Skeleton } from './Skeleton'

interface MapSkeletonProps {
  showControls?: boolean
  showMarkers?: boolean
  markerCount?: number
}

// Pre-computed marker positions to avoid Math.random during render
const MARKER_POSITIONS = [
  { top: 25, left: 20 },
  { top: 45, left: 55 },
  { top: 65, left: 35 },
  { top: 30, left: 75 },
  { top: 55, left: 15 },
  { top: 70, left: 60 },
  { top: 40, left: 40 },
  { top: 35, left: 80 },
  { top: 60, left: 25 },
  { top: 50, left: 70 },
]

export function MapSkeleton({
  showControls = true,
  showMarkers = true,
  markerCount = 5,
}: MapSkeletonProps) {
  const markerPositions = useMemo(() => MARKER_POSITIONS.slice(0, markerCount), [markerCount])

  return (
    <div className="relative w-full h-full min-h-[400px] bg-neutral-100 rounded-xl overflow-hidden">
      {/* Map background pattern */}
      <div className="absolute inset-0">
        {/* Grid pattern to simulate map tiles */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(to right, #e5e7eb 1px, transparent 1px),
              linear-gradient(to bottom, #e5e7eb 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px',
          }}
        />

        {/* Simulated roads */}
        <div className="absolute inset-0">
          <div className="absolute top-1/3 left-0 right-0 h-2 bg-neutral-200" />
          <div className="absolute top-2/3 left-0 right-0 h-1 bg-neutral-200" />
          <div className="absolute left-1/4 top-0 bottom-0 w-2 bg-neutral-200" />
          <div className="absolute left-2/3 top-0 bottom-0 w-1 bg-neutral-200" />
        </div>
      </div>

      {/* Placeholder markers */}
      {showMarkers && (
        <div className="absolute inset-0">
          {markerPositions.map((pos, i) => (
            <div
              key={i}
              className="absolute animate-pulse"
              style={{
                top: `${pos.top}%`,
                left: `${pos.left}%`,
              }}
            >
              <div className="relative">
                <MapPin className="w-6 h-6 text-neutral-400" />
                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-neutral-300 rounded-full" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Zoom controls skeleton */}
      {showControls && (
        <div className="absolute top-4 right-4 flex flex-col gap-1">
          <Skeleton className="w-8 h-8 rounded-lg" />
          <Skeleton className="w-8 h-8 rounded-lg" />
        </div>
      )}

      {/* Loading indicator */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="bg-white/90 backdrop-blur-sm rounded-xl px-6 py-4 shadow-lg flex items-center gap-3">
          <div className="relative">
            <Map className="w-6 h-6 text-primary-500" />
            <div className="absolute inset-0 animate-ping">
              <Map className="w-6 h-6 text-primary-500 opacity-50" />
            </div>
          </div>
          <div className="space-y-1">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-3 w-16" />
          </div>
        </div>
      </div>

      {/* Shimmer effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
    </div>
  )
}

// Mini map skeleton for sidebars
export function MiniMapSkeleton() {
  return (
    <div className="relative w-full aspect-video bg-neutral-100 rounded-lg overflow-hidden">
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(to right, #e5e7eb 1px, transparent 1px),
            linear-gradient(to bottom, #e5e7eb 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px',
        }}
      />
      <div className="absolute inset-0 flex items-center justify-center">
        <Map className="w-8 h-8 text-neutral-300 animate-pulse" />
      </div>
    </div>
  )
}
