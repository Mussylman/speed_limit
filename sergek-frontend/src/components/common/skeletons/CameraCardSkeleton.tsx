/**
 * Camera Card Skeleton
 * Loading placeholder for camera cards
 * Requirements: 8.3 - Camera card skeleton
 */

import { Skeleton } from './Skeleton'

interface CameraCardSkeletonProps {
  showOverlay?: boolean
}

export function CameraCardSkeleton({ showOverlay = true }: CameraCardSkeletonProps) {
  return (
    <div className="relative overflow-hidden rounded-xl bg-neutral-900">
      {/* Video area skeleton */}
      <div className="relative aspect-video">
        <Skeleton className="absolute inset-0 rounded-none bg-neutral-800" />

        {/* Status indicator skeleton - top right */}
        <div className="absolute top-3 right-3 z-20">
          <Skeleton className="w-20 h-6 rounded-full bg-neutral-700" />
        </div>

        {/* Smart badge skeleton - top left */}
        <div className="absolute top-3 left-3 z-20">
          <Skeleton className="w-14 h-6 rounded-full bg-neutral-700" />
        </div>

        {/* Bottom overlay skeleton */}
        {showOverlay && (
          <div className="absolute bottom-0 left-0 right-0 z-20 p-4">
            <div className="flex items-end justify-between gap-4">
              <div className="flex-1 min-w-0 space-y-2">
                <Skeleton className="h-5 w-3/4 bg-neutral-700" />
                <div className="flex items-center gap-1.5">
                  <Skeleton className="w-3.5 h-3.5 rounded-full bg-neutral-700" />
                  <Skeleton className="h-3 w-1/2 bg-neutral-700" />
                </div>
              </div>
              <Skeleton className="w-12 h-6 rounded bg-neutral-700" />
            </div>
          </div>
        )}

        {/* Shimmer overlay effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-neutral-700/20 to-transparent animate-shimmer" />
      </div>
    </div>
  )
}

// Grid of camera card skeletons
interface CameraGridSkeletonProps {
  count?: number
}

export function CameraGridSkeleton({ count = 15 }: CameraGridSkeletonProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="animate-fadeIn" style={{ animationDelay: `${i * 50}ms` }}>
          <CameraCardSkeleton />
        </div>
      ))}
    </div>
  )
}
