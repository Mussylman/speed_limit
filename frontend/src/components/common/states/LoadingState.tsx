/**
 * LoadingState Component
 * Full-page and inline loading indicators
 * Requirements: 8.3 - Loading feedback
 */

import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Loader2 } from 'lucide-react'
import { timing, easing } from '../animations/variants'

interface LoadingStateProps {
  message?: string
  size?: 'sm' | 'md' | 'lg'
  fullPage?: boolean
  className?: string
}

export function LoadingState({
  message,
  size = 'md',
  fullPage = false,
  className = '',
}: LoadingStateProps) {
  const { t } = useTranslation()
  const displayMessage = message || t('common.loading', { defaultValue: 'Loading...' })

  const sizeConfig = {
    sm: { icon: 'w-5 h-5', text: 'text-sm', gap: 'gap-2' },
    md: { icon: 'w-8 h-8', text: 'text-base', gap: 'gap-3' },
    lg: { icon: 'w-12 h-12', text: 'text-lg', gap: 'gap-4' },
  }

  const config = sizeConfig[size]

  const content = (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: timing.fast }}
      className={`flex flex-col items-center justify-center ${config.gap} ${className}`}
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      >
        <Loader2 className={`${config.icon} text-primary-500`} />
      </motion.div>
      <span className={`${config.text} text-neutral-500 font-medium`}>{displayMessage}</span>
    </motion.div>
  )

  if (fullPage) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/90 backdrop-blur-sm">
        {content}
      </div>
    )
  }

  return content
}

// Inline spinner
interface SpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg'
  color?: 'primary' | 'white' | 'neutral'
  className?: string
}

export function Spinner({ size = 'md', color = 'primary', className = '' }: SpinnerProps) {
  const sizeClasses = {
    xs: 'w-3 h-3',
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  }

  const colorClasses = {
    primary: 'text-primary-500',
    white: 'text-white',
    neutral: 'text-neutral-400',
  }

  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      className={className}
    >
      <Loader2 className={`${sizeClasses[size]} ${colorClasses[color]}`} />
    </motion.div>
  )
}

// Loading dots animation
interface LoadingDotsProps {
  color?: 'primary' | 'white' | 'neutral'
  className?: string
}

export function LoadingDots({ color = 'primary', className = '' }: LoadingDotsProps) {
  const colorClasses = {
    primary: 'bg-primary-500',
    white: 'bg-white',
    neutral: 'bg-neutral-400',
  }

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          animate={{
            y: [0, -4, 0],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: i * 0.1,
            ease: 'easeInOut',
          }}
          className={`w-1.5 h-1.5 rounded-full ${colorClasses[color]}`}
        />
      ))}
    </div>
  )
}

// Progress bar loading
interface ProgressLoadingProps {
  progress?: number
  indeterminate?: boolean
  label?: string
  className?: string
}

export function ProgressLoading({
  progress = 0,
  indeterminate = false,
  label,
  className = '',
}: ProgressLoadingProps) {
  return (
    <div className={`w-full ${className}`}>
      {label && <p className="text-sm text-neutral-600 mb-2">{label}</p>}
      <div className="h-2 bg-neutral-200 rounded-full overflow-hidden">
        {indeterminate ? (
          <motion.div
            animate={{ x: ['-100%', '100%'] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            className="h-full w-1/3 bg-primary-500 rounded-full"
          />
        ) : (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: timing.normal, ease: easing.decelerate }}
            className="h-full bg-primary-500 rounded-full"
          />
        )}
      </div>
      {!indeterminate && progress > 0 && (
        <p className="text-xs text-neutral-500 mt-1 text-right">{Math.round(progress)}%</p>
      )}
    </div>
  )
}

// Skeleton pulse overlay
interface LoadingOverlayProps {
  isLoading: boolean
  children: React.ReactNode
  className?: string
}

export function LoadingOverlay({ isLoading, children, className = '' }: LoadingOverlayProps) {
  return (
    <div className={`relative ${className}`}>
      {children}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center rounded-inherit"
        >
          <Spinner size="md" />
        </motion.div>
      )}
    </div>
  )
}
