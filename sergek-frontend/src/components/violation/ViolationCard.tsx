/**
 * ViolationCard Component - Violation summary card
 * Requirements: 6.2, 6.3 - Display violation info with thumbnail and quick status change
 *
 * Aesthetic: Industrial card with status-driven visual hierarchy
 */

import { useTranslation } from 'react-i18next'
import {
  Clock,
  Camera,
  Eye,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Car,
  Ban,
  Phone,
  ParkingCircle,
  CircleDot,
} from 'lucide-react'
import { PlateDisplay } from '../plate/PlateDisplay'
import type { Violation, ViolationType, ViolationStatus } from '../../types'

export interface ViolationCardProps {
  violation: Violation
  onStatusChange: (id: string, status: ViolationStatus) => void
  onViewDetail: (id: string) => void
}

const statusConfig: Record<
  ViolationStatus,
  { color: string; bgColor: string; icon: typeof AlertCircle }
> = {
  pending: {
    color: 'text-warning-600',
    bgColor: 'bg-warning-50 border-warning-200',
    icon: AlertCircle,
  },
  confirmed: {
    color: 'text-error-600',
    bgColor: 'bg-error-50 border-error-200',
    icon: CheckCircle,
  },
  dismissed: {
    color: 'text-neutral-500',
    bgColor: 'bg-neutral-50 border-neutral-200',
    icon: XCircle,
  },
}

const typeIcons: Record<ViolationType, typeof Zap> = {
  speed_limit: Zap,
  red_light: Ban,
  wrong_lane: Car,
  no_seatbelt: CircleDot,
  phone_usage: Phone,
  parking: ParkingCircle,
  other: AlertCircle,
}

function formatDate(date: Date): string {
  return date.toLocaleDateString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  })
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('ru-RU', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatFine(amount: number | undefined): string {
  if (!amount) return '—'
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'KZT',
    maximumFractionDigits: 0,
  }).format(amount)
}

export function ViolationCard({ violation, onStatusChange, onViewDetail }: ViolationCardProps) {
  const { t } = useTranslation()
  const { bgColor, icon: StatusIcon } = statusConfig[violation.status]
  const TypeIcon = typeIcons[violation.type]

  const handleStatusClick = (newStatus: ViolationStatus) => {
    if (violation.status !== newStatus) {
      onStatusChange(violation.id, newStatus)
    }
  }

  return (
    <div className={`card overflow-hidden border ${bgColor}`}>
      {/* Image Thumbnail */}
      <div className="relative h-40 bg-neutral-200 overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center bg-neutral-100">
          <Camera className="w-12 h-12 text-neutral-300" />
        </div>
        {/* Violation type badge */}
        <div className="absolute top-3 left-3">
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-neutral-900/80 backdrop-blur-sm rounded-full">
            <TypeIcon className="w-3.5 h-3.5 text-white" />
            <span className="text-xs font-medium text-white">
              {t(`violation.types.${violation.type}`)}
            </span>
          </div>
        </div>
        {/* Status badge */}
        <div className="absolute top-3 right-3">
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${
              violation.status === 'pending'
                ? 'bg-warning-500'
                : violation.status === 'confirmed'
                  ? 'bg-error-500'
                  : 'bg-neutral-500'
            }`}
          >
            <StatusIcon className="w-3.5 h-3.5 text-white" />
            <span className="text-xs font-medium text-white">
              {t(`violation.${violation.status}`)}
            </span>
          </div>
        </div>
        {/* Fine amount */}
        {violation.fine && (
          <div className="absolute bottom-3 right-3">
            <div className="px-3 py-1.5 bg-error-600 text-white text-sm font-bold rounded-lg shadow-lg">
              {formatFine(violation.fine)}
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Plate */}
        <div className="mb-3">
          <PlateDisplay plate={violation.plate} size="small" />
        </div>

        {/* Meta info */}
        <div className="flex items-center gap-4 text-sm text-neutral-500 mb-3">
          <div className="flex items-center gap-1.5">
            <Clock className="w-4 h-4" />
            <span>{formatDate(violation.timestamp)}</span>
            <span className="text-neutral-300">•</span>
            <span>{formatTime(violation.timestamp)}</span>
          </div>
        </div>

        {/* Description */}
        {violation.description && (
          <p className="body-sm text-neutral-600 mb-4 line-clamp-2">{violation.description}</p>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-3 border-t border-neutral-200">
          {/* Quick status buttons */}
          <div className="flex items-center gap-1 flex-1">
            <button
              onClick={() => handleStatusClick('confirmed')}
              disabled={violation.status === 'confirmed'}
              className={`p-2 rounded-lg transition-colors ${
                violation.status === 'confirmed'
                  ? 'bg-error-100 text-error-600'
                  : 'hover:bg-error-50 text-neutral-400 hover:text-error-500'
              }`}
              title={t('violation.confirmed')}
            >
              <CheckCircle className="w-5 h-5" />
            </button>
            <button
              onClick={() => handleStatusClick('dismissed')}
              disabled={violation.status === 'dismissed'}
              className={`p-2 rounded-lg transition-colors ${
                violation.status === 'dismissed'
                  ? 'bg-neutral-200 text-neutral-600'
                  : 'hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600'
              }`}
              title={t('violation.dismissed')}
            >
              <XCircle className="w-5 h-5" />
            </button>
          </div>

          {/* View detail button */}
          <button onClick={() => onViewDetail(violation.id)} className="btn btn--primary btn--sm">
            <Eye className="w-4 h-4" />
            {t('violations.viewDetail')}
          </button>
        </div>
      </div>
    </div>
  )
}
