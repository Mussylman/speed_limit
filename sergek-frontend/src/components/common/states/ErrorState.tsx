/**
 * ErrorState Component
 * Displays error messages with retry option
 * Requirements: 1.5 - Error state display
 */

import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import {
  AlertTriangle,
  RefreshCw,
  XCircle,
  ServerCrash,
  WifiOff,
  FileWarning,
  type LucideIcon,
} from 'lucide-react'
import { timing, easing } from '../animations/variants'

type ErrorType = 'generic' | 'network' | 'server' | 'notFound' | 'permission'

interface ErrorStateProps {
  type?: ErrorType
  title?: string
  message?: string
  error?: Error | string
  icon?: LucideIcon
  onRetry?: () => void
  retryLabel?: string
  className?: string
}

const errorConfig: Record<
  ErrorType,
  { icon: LucideIcon; titleKey: string; messageKey: string; color: string }
> = {
  generic: {
    icon: AlertTriangle,
    titleKey: 'errors.generic.title',
    messageKey: 'errors.generic.message',
    color: 'text-warning-500',
  },
  network: {
    icon: WifiOff,
    titleKey: 'errors.network.title',
    messageKey: 'errors.network.message',
    color: 'text-error-500',
  },
  server: {
    icon: ServerCrash,
    titleKey: 'errors.server.title',
    messageKey: 'errors.server.message',
    color: 'text-error-500',
  },
  notFound: {
    icon: FileWarning,
    titleKey: 'errors.notFound.title',
    messageKey: 'errors.notFound.message',
    color: 'text-warning-500',
  },
  permission: {
    icon: XCircle,
    titleKey: 'errors.permission.title',
    messageKey: 'errors.permission.message',
    color: 'text-error-500',
  },
}

export function ErrorState({
  type = 'generic',
  title,
  message,
  error,
  icon,
  onRetry,
  retryLabel,
  className = '',
}: ErrorStateProps) {
  const { t } = useTranslation()
  const config = errorConfig[type]
  const Icon = icon || config.icon

  const displayTitle = title || t(config.titleKey, { defaultValue: 'Something went wrong' })
  const displayMessage =
    message ||
    (error ? (typeof error === 'string' ? error : error.message) : null) ||
    t(config.messageKey, { defaultValue: 'An unexpected error occurred. Please try again.' })
  const displayRetryLabel = retryLabel || t('common.retry', { defaultValue: 'Try again' })

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: timing.normal, ease: easing.decelerate }}
      className={`flex flex-col items-center justify-center py-16 px-6 ${className}`}
    >
      {/* Icon with error styling */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: timing.normal, delay: 0.1, ease: easing.bounce }}
        className="relative mb-6"
      >
        <div className="w-20 h-20 bg-error-50 rounded-full flex items-center justify-center">
          <Icon className={`w-10 h-10 ${config.color}`} />
        </div>

        {/* Pulsing ring for attention */}
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.5, 0.2, 0.5],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="absolute inset-0 -m-2 border-2 border-error-200 rounded-full"
        />
      </motion.div>

      {/* Title */}
      <motion.h3
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: timing.normal, delay: 0.15, ease: easing.decelerate }}
        className="text-lg font-semibold text-neutral-800 mb-2 text-center font-display"
      >
        {displayTitle}
      </motion.h3>

      {/* Message */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: timing.normal, delay: 0.2, ease: easing.decelerate }}
        className="text-sm text-neutral-500 text-center max-w-sm mb-6"
      >
        {displayMessage}
      </motion.p>

      {/* Retry button */}
      {onRetry && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: timing.normal, delay: 0.25, ease: easing.decelerate }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-neutral-900 text-white rounded-lg font-medium text-sm hover:bg-neutral-800 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          {displayRetryLabel}
        </motion.button>
      )}
    </motion.div>
  )
}

// Inline error message
interface InlineErrorProps {
  message: string
  onRetry?: () => void
  className?: string
}

export function InlineError({ message, onRetry, className = '' }: InlineErrorProps) {
  return (
    <div
      className={`flex items-center justify-between gap-4 p-4 bg-error-50 border border-error-200 rounded-lg ${className}`}
    >
      <div className="flex items-center gap-3">
        <AlertTriangle className="w-5 h-5 text-error-500 flex-shrink-0" />
        <span className="text-sm text-error-700">{message}</span>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex-shrink-0 text-sm font-medium text-error-600 hover:text-error-700 flex items-center gap-1"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      )}
    </div>
  )
}
