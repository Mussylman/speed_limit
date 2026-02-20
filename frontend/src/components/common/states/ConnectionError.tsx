/**
 * ConnectionError Component
 * Specialized error state for connection/network issues
 * Requirements: 1.5 - Connection error display
 */

import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { WifiOff, RefreshCw, Signal, SignalLow, SignalZero } from 'lucide-react'
import { timing, easing } from '../animations/variants'

interface ConnectionErrorProps {
  type?: 'camera' | 'network' | 'server'
  cameraName?: string
  onRetry?: () => void
  onDismiss?: () => void
  isRetrying?: boolean
  className?: string
}

export function ConnectionError({
  type = 'network',
  cameraName,
  onRetry,
  onDismiss,
  isRetrying = false,
  className = '',
}: ConnectionErrorProps) {
  const { t } = useTranslation()

  const config = {
    camera: {
      icon: SignalZero,
      title: t('errors.cameraConnection.title', { defaultValue: 'Camera Disconnected' }),
      message: cameraName
        ? t('errors.cameraConnection.messageWithName', {
            name: cameraName,
            defaultValue: `Unable to connect to ${cameraName}. The camera may be offline or experiencing issues.`,
          })
        : t('errors.cameraConnection.message', {
            defaultValue: 'Unable to connect to the camera. Please check the connection.',
          }),
    },
    network: {
      icon: WifiOff,
      title: t('errors.network.title', { defaultValue: 'Connection Lost' }),
      message: t('errors.network.message', {
        defaultValue: 'Unable to connect to the server. Please check your internet connection.',
      }),
    },
    server: {
      icon: SignalLow,
      title: t('errors.server.title', { defaultValue: 'Server Unavailable' }),
      message: t('errors.server.message', {
        defaultValue: 'The server is temporarily unavailable. Please try again later.',
      }),
    },
  }

  const { icon: Icon, title, message } = config[type]

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: timing.normal, ease: easing.decelerate }}
      className={`relative overflow-hidden rounded-xl bg-neutral-900 ${className}`}
    >
      {/* Background pattern */}
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

      {/* Content */}
      <div className="relative flex flex-col items-center justify-center py-12 px-6">
        {/* Animated icon */}
        <motion.div
          animate={
            isRetrying
              ? { rotate: 360 }
              : {
                  scale: [1, 1.05, 1],
                  opacity: [0.8, 1, 0.8],
                }
          }
          transition={
            isRetrying
              ? { duration: 1, repeat: Infinity, ease: 'linear' }
              : { duration: 2, repeat: Infinity, ease: 'easeInOut' }
          }
          className="mb-4"
        >
          {isRetrying ? (
            <RefreshCw className="w-12 h-12 text-neutral-400" />
          ) : (
            <Icon className="w-12 h-12 text-error-400" />
          )}
        </motion.div>

        {/* Signal bars animation */}
        {!isRetrying && (
          <div className="flex items-end gap-1 mb-4">
            {[1, 2, 3, 4].map((bar) => (
              <motion.div
                key={bar}
                animate={{
                  opacity: [0.3, 0.6, 0.3],
                  height: [`${bar * 4}px`, `${bar * 6}px`, `${bar * 4}px`],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  delay: bar * 0.1,
                  ease: 'easeInOut',
                }}
                className="w-1.5 bg-error-400 rounded-full"
                style={{ height: `${bar * 4}px` }}
              />
            ))}
          </div>
        )}

        {/* Title */}
        <h3 className="text-lg font-semibold text-white mb-2 text-center font-display">
          {isRetrying ? t('common.reconnecting', { defaultValue: 'Reconnecting...' }) : title}
        </h3>

        {/* Message */}
        <p className="text-sm text-neutral-400 text-center max-w-xs mb-6">{message}</p>

        {/* Actions */}
        <div className="flex items-center gap-3">
          {onRetry && !isRetrying && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onRetry}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white text-neutral-900 rounded-lg font-medium text-sm hover:bg-neutral-100 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              {t('common.retry', { defaultValue: 'Retry' })}
            </motion.button>
          )}
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="px-4 py-2 text-neutral-400 hover:text-white text-sm font-medium transition-colors"
            >
              {t('common.dismiss', { defaultValue: 'Dismiss' })}
            </button>
          )}
        </div>
      </div>

      {/* Scanning line effect */}
      <motion.div
        animate={{ y: ['0%', '100%', '0%'] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-error-500/50 to-transparent"
      />
    </motion.div>
  )
}

// Compact connection status indicator
interface ConnectionStatusProps {
  status: 'connected' | 'connecting' | 'disconnected'
  label?: string
  className?: string
}

export function ConnectionStatus({ status, label, className = '' }: ConnectionStatusProps) {
  const { t } = useTranslation()

  const statusConfig = {
    connected: {
      icon: Signal,
      color: 'text-success-500',
      bgColor: 'bg-success-100',
      label: t('status.connected', { defaultValue: 'Connected' }),
    },
    connecting: {
      icon: SignalLow,
      color: 'text-warning-500',
      bgColor: 'bg-warning-100',
      label: t('status.connecting', { defaultValue: 'Connecting...' }),
    },
    disconnected: {
      icon: SignalZero,
      color: 'text-error-500',
      bgColor: 'bg-error-100',
      label: t('status.disconnected', { defaultValue: 'Disconnected' }),
    },
  }

  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bgColor} ${className}`}
    >
      <motion.div
        animate={
          status === 'connecting'
            ? { opacity: [1, 0.5, 1] }
            : status === 'connected'
              ? { scale: [1, 1.1, 1] }
              : {}
        }
        transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
      >
        <Icon className={`w-4 h-4 ${config.color}`} />
      </motion.div>
      <span className={`text-xs font-medium ${config.color}`}>{label || config.label}</span>
    </div>
  )
}
