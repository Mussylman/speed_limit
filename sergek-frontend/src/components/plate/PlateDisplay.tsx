/**
 * PlateDisplay Component - Kazakhstan License Plate Style
 * Requirements: 2.3 - Display recognized plates in large format
 *
 * Aesthetic: Industrial-utilitarian with Kazakhstan's distinctive yellow plates
 * Features: 4 size variants, animated entrance effect
 */

import { useEffect, useState } from 'react'
import { cn } from '../../utils/cn'

export type PlateSize = 'small' | 'medium' | 'large' | 'hero'

export interface PlateDisplayProps {
  plate: string
  size?: PlateSize
  animated?: boolean
  className?: string
  onClick?: () => void
}

const sizeClasses: Record<PlateSize, string> = {
  small: 'plate-display--small text-lg py-1.5 px-3 border-2',
  medium: 'plate-display--medium text-2xl py-2 px-4',
  large: 'plate-display--large text-4xl py-3 px-6',
  hero: 'plate-display--hero text-5xl md:text-6xl lg:text-7xl py-4 px-8 rounded-xl',
}

/**
 * Formats a Kazakhstan license plate for display
 * Standard format: XXX 000 XX (region code + numbers + letters)
 */
function formatPlate(plate: string): string {
  // Remove any existing spaces and convert to uppercase
  const cleaned = plate.replace(/\s/g, '').toUpperCase()
  return cleaned
}

export function PlateDisplay({
  plate,
  size = 'medium',
  animated = false,
  className,
  onClick,
}: PlateDisplayProps) {
  const [isVisible, setIsVisible] = useState(!animated)

  useEffect(() => {
    if (animated) {
      // Small delay for animation trigger
      const timer = setTimeout(() => setIsVisible(true), 50)
      return () => clearTimeout(timer)
    }
  }, [animated])

  const formattedPlate = formatPlate(plate)

  return (
    <div
      className={cn(
        'plate-display',
        sizeClasses[size],
        animated && 'plate-display--animated',
        !isVisible && animated && 'opacity-0',
        onClick && 'cursor-pointer hover:scale-105 transition-transform',
        className
      )}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onClick()
              }
            }
          : undefined
      }
      aria-label={onClick ? `License plate ${formattedPlate}` : undefined}
    >
      <span className="relative z-10 select-none">{formattedPlate}</span>

      {/* Kazakhstan flag indicator for authenticity */}
      <span
        className={cn(
          'absolute top-1 left-1 flex items-center gap-0.5',
          size === 'small' && 'hidden',
          size === 'medium' && 'text-[8px]',
          size === 'large' && 'text-[10px]',
          size === 'hero' && 'text-xs top-2 left-2'
        )}
      >
        <span className="text-sky-500 font-bold">KZ</span>
      </span>
    </div>
  )
}
