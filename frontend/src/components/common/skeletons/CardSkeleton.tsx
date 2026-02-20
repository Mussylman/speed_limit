/**
 * Card Skeletons
 * Loading placeholders for various card types
 * Requirements: 8.3 - Loading states
 */

import { Skeleton, SkeletonCircle } from './Skeleton'

// Generic card skeleton
interface CardSkeletonProps {
  showImage?: boolean
  showAvatar?: boolean
  lines?: number
}

export function CardSkeleton({
  showImage = false,
  showAvatar = false,
  lines = 2,
}: CardSkeletonProps) {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 overflow-hidden">
      {showImage && <Skeleton className="w-full aspect-video rounded-none" />}
      <div className="p-4">
        <div className="flex items-start gap-3">
          {showAvatar && <SkeletonCircle size="md" />}
          <div className="flex-1 space-y-2">
            <Skeleton className="h-5 w-3/4" />
            {Array.from({ length: lines }).map((_, i) => (
              <Skeleton key={i} className={`h-4 ${i === lines - 1 ? 'w-1/2' : 'w-full'}`} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Stat card skeleton (for dashboard)
export function StatCardSkeleton() {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 p-6">
      <div className="flex items-start justify-between">
        <div className="space-y-3 flex-1">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-8 w-20" />
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-4 w-16" />
          </div>
        </div>
        <Skeleton className="w-12 h-12 rounded-xl" />
      </div>
    </div>
  )
}

// Violation card skeleton
export function ViolationCardSkeleton() {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 overflow-hidden">
      <div className="flex">
        {/* Thumbnail */}
        <Skeleton className="w-32 h-24 rounded-none" />

        {/* Content */}
        <div className="flex-1 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <Skeleton className="h-5 w-24" />
            <Skeleton className="h-5 w-16 rounded-full" />
          </div>
          <Skeleton className="h-4 w-3/4" />
          <div className="flex items-center gap-4">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-3 w-16" />
          </div>
        </div>
      </div>
    </div>
  )
}

// Plate detection card skeleton
export function PlateDetectionSkeleton() {
  return (
    <div className="flex items-center gap-4 p-3 bg-neutral-50 rounded-lg">
      {/* Plate skeleton */}
      <Skeleton className="w-28 h-8 rounded-md bg-amber-200" />

      {/* Info */}
      <div className="flex-1 space-y-1">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-3 w-16" />
      </div>

      {/* Confidence */}
      <Skeleton className="w-12 h-6 rounded-full" />
    </div>
  )
}

// Vehicle info card skeleton
export function VehicleInfoSkeleton() {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 p-6 space-y-4">
      {/* Header with plate */}
      <div className="flex items-center gap-4">
        <Skeleton className="w-36 h-12 rounded-lg bg-amber-200" />
        <div className="space-y-2">
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-4 w-24" />
        </div>
      </div>

      {/* Details grid */}
      <div className="grid grid-cols-2 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="space-y-1">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-5 w-24" />
          </div>
        ))}
      </div>
    </div>
  )
}

// Dashboard stats grid skeleton
export function DashboardStatsSkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <StatCardSkeleton key={i} />
      ))}
    </div>
  )
}
