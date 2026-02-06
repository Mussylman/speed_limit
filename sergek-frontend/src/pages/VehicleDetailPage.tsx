/**
 * VehicleDetailPage - Vehicle Information Display
 * Requirements: 3.2, 3.3 - Display vehicle details, owner info, detection history
 */

import { useParams, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useEffect, useState, useMemo } from 'react'
import {
  ArrowLeft,
  Car,
  User,
  Phone,
  MapPin,
  Calendar,
  Palette,
  Camera,
  Clock,
  Route,
  ChevronRight,
} from 'lucide-react'
import { PlateDisplay } from '../components/plate'
import { useVehicleStore } from '../stores/vehicleStore'
import { useCameraStore } from '../stores/cameraStore'
import { LoadingState, ErrorState, EmptyState } from '../components/common/states'
import { vehicleService } from '../services/api'
import type { PlateDetection, Vehicle } from '../types'

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
    second: '2-digit',
  })
}

function formatRelativeTime(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffMins < 1) return 'только что'
  if (diffMins < 60) return `${diffMins} мин. назад`
  if (diffHours < 24) return `${diffHours} ч. назад`
  return `${diffDays} дн. назад`
}

export function VehicleDetailPage() {
  const { plate } = useParams<{ plate: string }>()
  const { t } = useTranslation()
  const navigate = useNavigate()

  const { getVehicleByPlate, getDetectionsByPlate } = useVehicleStore()
  const getCameraById = useCameraStore((state) => state.getCameraById)

  const [vehicle, setVehicle] = useState<Vehicle | null>(null)
  const [detections, setDetections] = useState<PlateDetection[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load vehicle and detection data
  useEffect(() => {
    const loadVehicleData = async () => {
      if (!plate) return

      setLoading(true)
      setError(null)

      try {
        // Try to get from store first
        const storeVehicle = getVehicleByPlate(plate)
        const storeDetections = getDetectionsByPlate(plate)

        if (storeVehicle) {
          setVehicle(storeVehicle)
          setDetections(storeDetections)
        } else {
          // Load from API
          const [vehicleData, detectionsData] = await Promise.all([
            vehicleService.getByPlate(plate),
            vehicleService.getDetectionsByPlate(plate)
          ])

          setVehicle(vehicleData)
          setDetections(detectionsData)
        }
      } catch (err) {
        console.error('Failed to load vehicle data:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    loadVehicleData()
  }, [plate, getVehicleByPlate, getDetectionsByPlate, t])

  // Sort detections by timestamp (newest first)
  const sortedDetections = useMemo(
    () => [...detections].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()),
    [detections]
  )

  const handleShowRoute = () => {
    if (plate) {
      navigate(`/map?plate=${plate}`)
    }
  }

  if (!plate) {
    return (
      <ErrorState
        message={t('common.error')}
        onRetry={() => navigate('/vehicles')}
      />
    )
  }

  if (loading) {
    return <LoadingState message={t('vehicle.loading')} />
  }

  if (error) {
    return (
      <ErrorState
        message={error}
        onRetry={() => window.location.reload()}
      />
    )
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header with back button */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate(-1)}
          className="btn btn--ghost p-2 rounded-full"
          aria-label="Go back"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="heading-2">{t('pages.vehicleDetail.title')}</h1>
          <p className="body-sm mt-1">{t('pages.vehicleDetail.description')}</p>
        </div>
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Vehicle info */}
        <div className="lg:col-span-2 space-y-6">
          {/* Plate display card */}
          <div className="card p-6">
            <div className="flex flex-col sm:flex-row items-center gap-6">
              <PlateDisplay plate={plate} size="large" animated />
              <div className="flex-1 text-center sm:text-left">
                {vehicle ? (
                  <>
                    <h2 className="heading-3">
                      {vehicle.brand} {vehicle.model}
                    </h2>
                    <p className="body-base mt-1">
                      {vehicle.year} • {vehicle.color}
                    </p>
                  </>
                ) : (
                  <p className="body-base text-neutral-500">{t('common.noData')}</p>
                )}
              </div>
              <button
                onClick={handleShowRoute}
                className="btn btn--primary flex items-center gap-2"
              >
                <Route className="w-4 h-4" />
                <span>{t('vehicle.showRoute')}</span>
              </button>
            </div>
          </div>

          {/* Vehicle details card */}
          {vehicle && (
            <div className="card p-6">
              <h3 className="heading-4 mb-4 flex items-center gap-2">
                <Car className="w-5 h-5 text-primary-500" />
                {t('vehicle.details')}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <InfoItem
                  icon={<Car className="w-4 h-4" />}
                  label={t('vehicle.brand')}
                  value={vehicle.brand}
                />
                <InfoItem
                  icon={<Car className="w-4 h-4" />}
                  label={t('vehicle.model')}
                  value={vehicle.model}
                />
                <InfoItem
                  icon={<Palette className="w-4 h-4" />}
                  label={t('vehicle.color')}
                  value={vehicle.color}
                />
                <InfoItem
                  icon={<Calendar className="w-4 h-4" />}
                  label={t('vehicle.year')}
                  value={vehicle.year.toString()}
                />
              </div>
            </div>
          )}

          {/* Owner info card */}
          {vehicle?.owner && (
            <div className="card p-6">
              <h3 className="heading-4 mb-4 flex items-center gap-2">
                <User className="w-5 h-5 text-primary-500" />
                {t('vehicle.owner')}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <InfoItem
                  icon={<User className="w-4 h-4" />}
                  label={t('vehicle.ownerName')}
                  value={vehicle.owner.name}
                />
                <InfoItem
                  icon={<span className="text-xs font-mono">IIN</span>}
                  label={t('vehicle.iin')}
                  value={vehicle.owner.iin}
                  mono
                />
                {vehicle.owner.phone && (
                  <InfoItem
                    icon={<Phone className="w-4 h-4" />}
                    label={t('vehicle.phone')}
                    value={vehicle.owner.phone}
                  />
                )}
                {vehicle.owner.address && (
                  <InfoItem
                    icon={<MapPin className="w-4 h-4" />}
                    label={t('vehicle.address')}
                    value={vehicle.owner.address}
                  />
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right column - Detection history */}
        <div className="lg:col-span-1">
          <div className="card p-6 sticky top-6">
            <h3 className="heading-4 mb-4 flex items-center gap-2">
              <Camera className="w-5 h-5 text-primary-500" />
              {t('vehicle.detectionHistory')}
              <span className="badge badge--info ml-auto">{sortedDetections.length}</span>
            </h3>

            {sortedDetections.length === 0 ? (
              <EmptyState
                icon={Camera}
                title={t('plate.noDetections')}
                description=""
              />
            ) : (
              <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                {sortedDetections.map((detection, index) => (
                  <DetectionItem
                    key={detection.id}
                    detection={detection}
                    cameraName={getCameraById(detection.cameraId)?.name}
                    isFirst={index === 0}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Info item component for displaying labeled data
interface InfoItemProps {
  icon: React.ReactNode
  label: string
  value: string
  mono?: boolean
}

function InfoItem({ icon, label, value, mono }: InfoItemProps) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-neutral-50 hover:bg-neutral-100 transition-colors">
      <div className="text-neutral-400 mt-0.5">{icon}</div>
      <div className="min-w-0 flex-1">
        <p className="caption text-neutral-500">{label}</p>
        <p className={`body-base text-neutral-900 truncate ${mono ? 'font-mono' : ''}`}>{value}</p>
      </div>
    </div>
  )
}

// Detection history item component
interface DetectionItemProps {
  detection: PlateDetection
  cameraName?: string
  isFirst: boolean
}

function DetectionItem({ detection, cameraName, isFirst }: DetectionItemProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()

  return (
    <div
      className={`
        p-3 rounded-lg border transition-all cursor-pointer
        hover:border-primary-300 hover:bg-primary-50
        ${isFirst ? 'border-primary-200 bg-primary-50/50' : 'border-neutral-200 bg-white'}
      `}
      onClick={() => navigate(`/cameras/${detection.cameraId}`)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-neutral-400" />
          <span className="body-sm font-medium text-neutral-700">
            {cameraName || `Camera ${detection.cameraId}`}
          </span>
        </div>
        <ChevronRight className="w-4 h-4 text-neutral-400" />
      </div>
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-neutral-500">
            <Clock className="w-3 h-3" />
            {formatTime(detection.timestamp)}
          </span>
          <span className="text-neutral-400">{formatDate(detection.timestamp)}</span>
        </div>
        <span className="badge badge--neutral text-xs">
          {t('plate.lane')} {detection.lane}
        </span>
      </div>
      {isFirst && (
        <div className="mt-2 pt-2 border-t border-primary-200">
          <span className="caption text-primary-600">
            {formatRelativeTime(detection.timestamp)}
          </span>
        </div>
      )}
    </div>
  )
}
