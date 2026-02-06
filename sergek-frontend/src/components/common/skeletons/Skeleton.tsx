/**
 * Base Skeleton Components
 * Animated loading placeholders with shimmer effect
 * Requirements: 8.3 - Loading states
 */

import { cn } from '../../../utils/cn'

interface SkeletonProps {
  className?: string
  animate?: boolean
}

// Base skeleton with shimmer animation
export function Skeleton({ className = '', animate = true }: SkeletonProps) {
  return <div className={cn('bg-neutral-200 rounded', animate && 'animate-pulse', className)} />
}

// Text skeleton - for text lines
interface SkeletonTextProps extends SkeletonProps {
  lines?: number
  lastLineWidth?: 'full' | 'half' | 'three-quarters'
}

export function SkeletonText({
  className = '',
  animate = true,
  lines = 3,
  lastLineWidth = 'three-quarters',
}: SkeletonTextProps) {
  const lastLineWidthClass = {
    full: 'w-full',
    half: 'w-1/2',
    'three-quarters': 'w-3/4',
  }

  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          animate={animate}
          className={cn('h-4', i === lines - 1 ? lastLineWidthClass[lastLineWidth] : 'w-full')}
        />
      ))}
    </div>
  )
}

// Circle skeleton - for avatars, icons
interface SkeletonCircleProps extends SkeletonProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
}

export function SkeletonCircle({
  className = '',
  animate = true,
  size = 'md',
}: SkeletonCircleProps) {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
    xl: 'w-24 h-24',
  }

  return <Skeleton animate={animate} className={cn('rounded-full', sizeClasses[size], className)} />
}

// Image skeleton - for images with aspect ratio
interface SkeletonImageProps extends SkeletonProps {
  aspectRatio?: 'video' | 'square' | 'portrait' | 'wide'
}

export function SkeletonImage({
  className = '',
  animate = true,
  aspectRatio = 'video',
}: SkeletonImageProps) {
  const aspectClasses = {
    video: 'aspect-video',
    square: 'aspect-square',
    portrait: 'aspect-[3/4]',
    wide: 'aspect-[21/9]',
  }

  return (
    <Skeleton
      animate={animate}
      className={cn('w-full rounded-lg', aspectClasses[aspectRatio], className)}
    />
  )
}
