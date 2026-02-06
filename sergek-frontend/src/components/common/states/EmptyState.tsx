/**
 * EmptyState Component
 * Displays when no data is available
 * Requirements: 1.5 - Visual feedback for empty states
 */

import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import {
  Camera,
  Video,
  FileSearch,
  AlertCircle,
  Car,
  MapPin,
  Search,
  Plus,
  type LucideIcon,
} from 'lucide-react'
import { timing, easing } from '../animations/variants'

type EmptyStateType =
  | 'cameras'
  | 'detections'
  | 'violations'
  | 'vehicles'
  | 'search'
  | 'map'
  | 'generic'

interface EmptyStateProps {
  type?: EmptyStateType
  title?: string
  description?: string
  icon?: LucideIcon
  action?: {
    label: string
    onClick: () => void
    icon?: LucideIcon
  }
  className?: string
}

const emptyStateConfig: Record<
  EmptyStateType,
  { icon: LucideIcon; titleKey: string; descriptionKey: string }
> = {
  cameras: {
    icon: Camera,
    titleKey: 'emptyStates.cameras.title',
    descriptionKey: 'emptyStates.cameras.description',
  },
  detections: {
    icon: Video,
    titleKey: 'emptyStates.detections.title',
    descriptionKey: 'emptyStates.detections.description',
  },
  violations: {
    icon: AlertCircle,
    titleKey: 'emptyStates.violations.title',
    descriptionKey: 'emptyStates.violations.description',
  },
  vehicles: {
    icon: Car,
    titleKey: 'emptyStates.vehicles.title',
    descriptionKey: 'emptyStates.vehicles.description',
  },
  search: {
    icon: Search,
    titleKey: 'emptyStates.search.title',
    descriptionKey: 'emptyStates.search.description',
  },
  map: {
    icon: MapPin,
    titleKey: 'emptyStates.map.title',
    descriptionKey: 'emptyStates.map.description',
  },
  generic: {
    icon: FileSearch,
    titleKey: 'emptyStates.generic.title',
    descriptionKey: 'emptyStates.generic.description',
  },
}

export function EmptyState({
  type = 'generic',
  title,
  description,
  icon,
  action,
  className = '',
}: EmptyStateProps) {
  const { t } = useTranslation()
  const config = emptyStateConfig[type]
  const Icon = icon || config.icon
  const ActionIcon = action?.icon || Plus

  const displayTitle = title || t(config.titleKey, { defaultValue: 'No data found' })
  const displayDescription =
    description ||
    t(config.descriptionKey, { defaultValue: 'There is no data to display at this time.' })

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: timing.normal, ease: easing.decelerate }}
      className={`flex flex-col items-center justify-center py-16 px-6 ${className}`}
    >
      {/* Icon container with subtle animation */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: timing.normal, delay: 0.1, ease: easing.bounce }}
        className="relative mb-6"
      >
        {/* Background circle */}
        <div className="w-20 h-20 bg-neutral-100 rounded-full flex items-center justify-center">
          <Icon className="w-10 h-10 text-neutral-400" />
        </div>

        {/* Decorative rings */}
        <div className="absolute inset-0 -m-2 border-2 border-neutral-100 rounded-full opacity-50" />
        <div className="absolute inset-0 -m-4 border border-neutral-100 rounded-full opacity-25" />
      </motion.div>

      {/* Title */}
      <motion.h3
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: timing.normal, delay: 0.15, ease: easing.decelerate }}
        className="text-lg font-semibold text-neutral-700 mb-2 text-center font-display"
      >
        {displayTitle}
      </motion.h3>

      {/* Description */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: timing.normal, delay: 0.2, ease: easing.decelerate }}
        className="text-sm text-neutral-500 text-center max-w-sm mb-6"
      >
        {displayDescription}
      </motion.p>

      {/* Action button */}
      {action && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: timing.normal, delay: 0.25, ease: easing.decelerate }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={action.onClick}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary-500 text-white rounded-lg font-medium text-sm hover:bg-primary-600 transition-colors"
        >
          <ActionIcon className="w-4 h-4" />
          {action.label}
        </motion.button>
      )}
    </motion.div>
  )
}

// Compact empty state for inline use
interface CompactEmptyStateProps {
  message: string
  icon?: LucideIcon
  className?: string
}

export function CompactEmptyState({
  message,
  icon: Icon = FileSearch,
  className = '',
}: CompactEmptyStateProps) {
  return (
    <div className={`flex items-center justify-center gap-3 py-8 text-neutral-500 ${className}`}>
      <Icon className="w-5 h-5" />
      <span className="text-sm">{message}</span>
    </div>
  )
}
