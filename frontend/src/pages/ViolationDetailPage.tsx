/**
 * ViolationDetailPage - Detailed violation view
 * Requirements: 6.4, 6.6 - Display all violation info, evidence, and status update
 *
 * Aesthetic: Arctic Command Center - Evidence-focused layout with clear status controls
 */

import { useMemo, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  ArrowLeft,
  Camera,
  Clock,
  MapPin,
  Play,
  Image,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Car,
  Ban,
  Phone,
  ParkingCircle,
  CircleDot,
  History,
  DollarSign,
} from 'lucide-react'
import { useViolationStore } from '../stores/violationStore'
import { useCameraStore } from '../stores/cameraStore'
import { PlateDisplay } from '../components/plate/PlateDisplay'
import type { ViolationType, ViolationStatus } from '../types'

const statusConfig: Record<
  ViolationStatus,
  { color: string; bgColor: string; borderColor: string; icon: typeof AlertCircle; label: string }
> = {
  pending: {
    color: 'text-warning-600',
    bgColor: 'bg-warning-50',
    borderColor: 'border-warning-300',
    icon: AlertCircle,
    label: 'pending',
  },
  confirmed: {
    color: 'text-error-600',
    bgColor: 'bg-error-50',
    borderColor: 'border-error-300',
    icon: CheckCircle,
    label: 'confirmed',
  },
  dismissed: {
    color: 'text-neutral-500',
    bgColor: 'bg-neutral-50',
    borderColor: 'border-neutral-300',
    icon: XCircle,
    label: 'dismissed',
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

function formatDateTime(date: Date): string {
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
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

export function ViolationDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { getViolationById, updateViolationStatus } = useViolationStore()
  const { cameras } = useCameraStore()

  // Use useMemo to derive violation from store instead of useEffect + setState
  const violation = useMemo(() => {
    if (!id) return undefined
    return getViolationById(id)
  }, [id, getViolationById])

  const [showVideo, setShowVideo] = useState(false)

  if (!violation) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-neutral-300 mx-auto mb-4" />
          <p className="heading-4 text-neutral-500">{t('common.noData')}</p>
          <Link to="/violations" className="btn btn--primary mt-4">
            <ArrowLeft className="w-4 h-4" />
            {t('violations.backToList')}
          </Link>
        </div>
      </div>
    )
  }

  const { color, bgColor, borderColor, icon: StatusIcon } = statusConfig[violation.status]
  const TypeIcon = typeIcons[violation.type]
  const camera = cameras.find((c) => c.id === violation.cameraId)

  const handleStatusChange = (newStatus: ViolationStatus) => {
    updateViolationStatus(violation.id, newStatus)
  }

  return (
    <div className="h-full overflow-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button onClick={() => navigate('/violations')} className="btn btn--ghost">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h1 className="heading-2">{t('pages.violationDetail.title')}</h1>
          <p className="body-sm text-neutral-500">ID: {violation.id}</p>
        </div>
        {/* Status Badge */}
        <div
          className={`flex items-center gap-2 px-4 py-2 rounded-xl border-2 ${bgColor} ${borderColor}`}
        >
          <StatusIcon className={`w-5 h-5 ${color}`} />
          <span className={`font-semibold ${color}`}>{t(`violation.${violation.status}`)}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content - Evidence */}
        <div className="lg:col-span-2 space-y-6">
          {/* Evidence Image/Video */}
          <div className="card overflow-hidden">
            <div className="p-4 border-b border-neutral-200">
              <h3 className="heading-4 flex items-center gap-2">
                <Image className="w-5 h-5 text-primary-500" />
                {t('violations.evidence')}
              </h3>
            </div>
            <div className="relative aspect-video bg-neutral-900">
              {showVideo && violation.videoClipUrl ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <video
                    src={violation.videoClipUrl}
                    controls
                    autoPlay
                    className="w-full h-full object-contain"
                  />
                </div>
              ) : (
                <div className="absolute inset-0 flex items-center justify-center bg-neutral-800">
                  <Camera className="w-24 h-24 text-neutral-600" />
                </div>
              )}

              {/* Violation type overlay */}
              <div className="absolute top-4 left-4">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900/80 backdrop-blur-sm rounded-lg">
                  <TypeIcon className="w-4 h-4 text-white" />
                  <span className="text-sm font-medium text-white">
                    {t(`violation.types.${violation.type}`)}
                  </span>
                </div>
              </div>

              {/* Video toggle */}
              {violation.videoClipUrl && (
                <button
                  onClick={() => setShowVideo(!showVideo)}
                  className="absolute bottom-4 right-4 btn btn--primary"
                >
                  <Play className="w-4 h-4" />
                  {showVideo ? t('violations.evidence') : t('violations.videoClip')}
                </button>
              )}
            </div>
          </div>

          {/* Description */}
          {violation.description && (
            <div className="card p-4">
              <h3 className="heading-4 mb-3">{t('violations.description')}</h3>
              <p className="body-base text-neutral-700">{violation.description}</p>
            </div>
          )}

          {/* Status Update */}
          <div className="card p-4">
            <h3 className="heading-4 flex items-center gap-2 mb-4">
              <History className="w-5 h-5 text-primary-500" />
              {t('violations.updateStatus')}
            </h3>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => handleStatusChange('pending')}
                disabled={violation.status === 'pending'}
                className={`flex-1 min-w-[140px] flex items-center justify-center gap-2 px-4 py-3 rounded-xl border-2 transition-all ${
                  violation.status === 'pending'
                    ? 'bg-warning-100 border-warning-400 text-warning-700'
                    : 'border-neutral-200 hover:border-warning-300 hover:bg-warning-50 text-neutral-600'
                }`}
              >
                <AlertCircle className="w-5 h-5" />
                <span className="font-medium">{t('violation.pending')}</span>
              </button>
              <button
                onClick={() => handleStatusChange('confirmed')}
                disabled={violation.status === 'confirmed'}
                className={`flex-1 min-w-[140px] flex items-center justify-center gap-2 px-4 py-3 rounded-xl border-2 transition-all ${
                  violation.status === 'confirmed'
                    ? 'bg-error-100 border-error-400 text-error-700'
                    : 'border-neutral-200 hover:border-error-300 hover:bg-error-50 text-neutral-600'
                }`}
              >
                <CheckCircle className="w-5 h-5" />
                <span className="font-medium">{t('violation.confirmed')}</span>
              </button>
              <button
                onClick={() => handleStatusChange('dismissed')}
                disabled={violation.status === 'dismissed'}
                className={`flex-1 min-w-[140px] flex items-center justify-center gap-2 px-4 py-3 rounded-xl border-2 transition-all ${
                  violation.status === 'dismissed'
                    ? 'bg-neutral-200 border-neutral-400 text-neutral-700'
                    : 'border-neutral-200 hover:border-neutral-400 hover:bg-neutral-100 text-neutral-600'
                }`}
              >
                <XCircle className="w-5 h-5" />
                <span className="font-medium">{t('violation.dismissed')}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Sidebar - Details */}
        <div className="space-y-6">
          {/* Plate */}
          <div className="card p-4">
            <h3 className="body-sm text-neutral-500 mb-3">{t('violations.plateSearch')}</h3>
            <Link to={`/vehicles/${violation.plate}`}>
              <PlateDisplay plate={violation.plate} size="large" className="w-full" />
            </Link>
          </div>

          {/* Fine Amount */}
          {violation.fine && (
            <div className="card p-4 bg-error-50 border-error-200">
              <h3 className="body-sm text-error-600 flex items-center gap-2 mb-2">
                <DollarSign className="w-4 h-4" />
                {t('violations.fineAmount')}
              </h3>
              <p className="text-3xl font-bold text-error-700">{formatFine(violation.fine)}</p>
            </div>
          )}

          {/* Date & Time */}
          <div className="card p-4">
            <h3 className="body-sm text-neutral-500 flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4" />
              {t('violations.date')}
            </h3>
            <p className="heading-4">{formatDateTime(violation.timestamp)}</p>
          </div>

          {/* Camera Info */}
          <div className="card p-4">
            <h3 className="body-sm text-neutral-500 flex items-center gap-2 mb-2">
              <Camera className="w-4 h-4" />
              {t('violations.cameraInfo')}
            </h3>
            {camera ? (
              <div>
                <p className="heading-4">{camera.name}</p>
                <p className="body-sm text-neutral-500 flex items-center gap-1 mt-1">
                  <MapPin className="w-3.5 h-3.5" />
                  {camera.location.address}
                </p>
                <Link
                  to={`/cameras/${camera.id}`}
                  className="btn btn--secondary btn--sm mt-3 w-full"
                >
                  {t('map.viewStream')}
                </Link>
              </div>
            ) : (
              <p className="body-base text-neutral-600">{violation.cameraId}</p>
            )}
          </div>

          {/* Violation Type */}
          <div className="card p-4">
            <h3 className="body-sm text-neutral-500 mb-2">{t('violations.type')}</h3>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                <TypeIcon className="w-5 h-5 text-primary-600" />
              </div>
              <p className="heading-4">{t(`violation.types.${violation.type}`)}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
