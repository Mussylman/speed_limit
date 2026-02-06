/**
 * CameraDetailPage - Detailed camera view with live stream
 */

import { useParams, useNavigate, Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useEffect, useState } from 'react'
import {
  ArrowLeft,
  MapPin,
  Settings,
  Activity,
  Zap,
  Eye,
  Signal,
  AlertCircle,
} from 'lucide-react'
import { useCameraStore } from '../stores/cameraStore'
import { useVehicleStore } from '../stores/vehicleStore'
import { VideoPlayer } from '../components/camera/VideoPlayer'
import { PlateDetectionList } from '../components/plate'
import { LoadingState, ErrorState } from '../components/common/states'
import { cameraService, vehicleService } from '../services/api'
import type { Camera, PlateDetection } from '../types'

function formatDateTime(date: Date): string {
  return date.toLocaleString('tr-TR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function CameraDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { t } = useTranslation()
  const navigate = useNavigate()
  
  const { getCameraById } = useCameraStore()
  const { getDetectionsByCamera } = useVehicleStore()
  
  const [camera, setCamera] = useState<Camera | null>(null)
  const [detections, setDetections] = useState<PlateDetection[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [streamStatus, setStreamStatus] = useState<'loading' | 'playing' | 'paused' | 'error'>('loading')

  // Load camera and detection data
  useEffect(() => {
    const loadCameraData = async () => {
      if (!id) return

      setLoading(true)
      setError(null)

      try {
        // Try to get from store first
        const storeCamera = getCameraById(id)
        const storeDetections = getDetectionsByCamera(id)

        if (storeCamera) {
          setCamera(storeCamera)
          setDetections(storeDetections)
        } else {
          // Load from API
          const [cameraData, detectionsData] = await Promise.all([
            cameraService.getById(id),
            vehicleService.getDetectionsByCamera(id)
          ])

          setCamera(cameraData)
          setDetections(detectionsData)
        }
      } catch (err) {
        console.error('Failed to load camera data:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    loadCameraData()
  }, [id, getCameraById, getDetectionsByCamera, t])

  const handleStreamError = (error: string) => {
    console.error('Stream error:', error)
  }

  const handleStreamStatusChange = (status: typeof streamStatus) => {
    setStreamStatus(status)
  }

  if (!id) {
    return (
      <ErrorState
        message={t('common.error')}
        onRetry={() => navigate('/cameras')}
      />
    )
  }

  if (loading) {
    return <LoadingState message={t('camera.loading')} />
  }

  if (error || !camera) {
    return (
      <ErrorState
        message={error || t('camera.notFound')}
        onRetry={() => window.location.reload()}
      />
    )
  }

  const statusConfig: Record<Camera['status'], {
    color: string
    bgColor: string
    borderColor: string
    icon: typeof Activity
    label: string
  }> = {
    online: {
      color: 'text-success-600',
      bgColor: 'bg-success-50',
      borderColor: 'border-success-300',
      icon: Activity,
      label: t('camera.online')
    },
    offline: {
      color: 'text-neutral-500',
      bgColor: 'bg-neutral-50',
      borderColor: 'border-neutral-300',
      icon: AlertCircle,
      label: t('camera.offline')
    },
    error: {
      color: 'text-error-600',
      bgColor: 'bg-error-50',
      borderColor: 'border-error-300',
      icon: AlertCircle,
      label: t('camera.error')
    }
  }

  const { color, bgColor, borderColor, icon: StatusIcon, label } = statusConfig[camera.status]

  return (
    <div className="h-full overflow-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button onClick={() => navigate('/cameras')} className="btn btn--ghost">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h1 className="heading-2 flex items-center gap-3">
            {camera.type === 'smart' ? (
              <Zap className="w-8 h-8 text-warning-500" />
            ) : (
              <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                <div className="w-4 h-4 bg-primary-500 rounded" />
              </div>
            )}
            {camera.name}
          </h1>
          <p className="body-sm text-neutral-500 mt-1">ID: {camera.id}</p>
        </div>
        
        {/* Status Badge */}
        <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border-2 ${bgColor} ${borderColor}`}>
          <StatusIcon className={`w-5 h-5 ${color}`} />
          <span className={`font-semibold ${color}`}>{label}</span>
        </div>

        {/* Admin Actions */}
        <Link to={`/admin/cameras/${camera.id}/edit`} className="btn btn--secondary">
          <Settings className="w-4 h-4" />
          {t('common.edit')}
        </Link>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Main Content - Video Stream */}
        <div className="xl:col-span-2 space-y-6">
          {/* Video Player */}
          <div className="card overflow-hidden p-0">
            <div className="aspect-video">
              <VideoPlayer
                rtspUrl={camera.rtspUrl}
                hlsUrl={camera.hlsUrl}
                className="w-full h-full"
                autoPlay={camera.status === 'online'}
                onError={handleStreamError}
                onStatusChange={handleStreamStatusChange}
              />
            </div>
          </div>

          {/* Stream Info */}
          <div className="card p-6">
            <h3 className="heading-4 mb-4 flex items-center gap-2">
              <Signal className="w-5 h-5 text-primary-500" />
              {t('camera.streamInfo')}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-neutral-500">
                  RTSP URL
                </label>
                <div className="p-3 bg-neutral-50 rounded-lg font-mono text-sm break-all">
                  {camera.rtspUrl}
                </div>
              </div>
              <div className="space-y-2">
                <label className="block text-sm font-medium text-neutral-500">
                  HLS URL
                </label>
                <div className="p-3 bg-neutral-50 rounded-lg font-mono text-sm break-all">
                  {camera.hlsUrl}
                </div>
              </div>
            </div>
          </div>

          {/* Recent Detections (for smart cameras) */}
          {camera.type === 'smart' && (
            <div className="card p-6">
              <h3 className="heading-4 mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-primary-500" />
                {t('plate.recentDetections')}
                <span className="badge badge--info ml-auto">{detections.length}</span>
              </h3>
              <PlateDetectionList 
                detections={detections} 
                maxItems={10} 
                showCamera={false}
              />
            </div>
          )}
        </div>

        {/* Sidebar - Camera Details */}
        <div className="space-y-6">
          {/* Camera Info */}
          <div className="card p-6">
            <h3 className="heading-4 mb-4 flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-primary-500" />
              {t('camera.details')}
            </h3>
            <div className="space-y-4">
              <InfoItem
                label={t('camera.type')}
                value={
                  <div className="flex items-center gap-2">
                    {camera.type === 'smart' ? (
                      <>
                        <Zap className="w-4 h-4 text-warning-500" />
                        <span>{t('camera.smart')}</span>
                      </>
                    ) : (
                      <>
                        <div className="w-4 h-4 rounded bg-neutral-500" />
                        <span>{t('camera.standard')}</span>
                      </>
                    )}
                  </div>
                }
              />
              <InfoItem
                label={t('camera.status')}
                value={
                  <div className={`flex items-center gap-2 ${color}`}>
                    <StatusIcon className="w-4 h-4" />
                    <span>{label}</span>
                  </div>
                }
              />
              {camera.lane && (
                <InfoItem
                  label={t('camera.lane')}
                  value={camera.lane.toString()}
                />
              )}
              <InfoItem
                label={t('camera.streamStatus')}
                value={
                  <div className={`flex items-center gap-2 ${
                    streamStatus === 'playing' ? 'text-success-600' :
                    streamStatus === 'error' ? 'text-error-600' :
                    'text-neutral-500'
                  }`}>
                    <div className={`w-2 h-2 rounded-full ${
                      streamStatus === 'playing' ? 'bg-success-500' :
                      streamStatus === 'error' ? 'bg-error-500' :
                      'bg-neutral-400'
                    }`} />
                    <span className="capitalize">{streamStatus}</span>
                  </div>
                }
              />
            </div>
          </div>

          {/* Location Info */}
          <div className="card p-6">
            <h3 className="heading-4 mb-4 flex items-center gap-2">
              <MapPin className="w-5 h-5 text-primary-500" />
              {t('camera.location')}
            </h3>
            <div className="space-y-4">
              {camera.location.address && (
                <InfoItem
                  label={t('camera.address')}
                  value={camera.location.address}
                />
              )}
              <InfoItem
                label={t('camera.coordinates')}
                value={`${camera.location.lat.toFixed(6)}, ${camera.location.lng.toFixed(6)}`}
                mono
              />
              <Link 
                to={`/map?camera=${camera.id}`}
                className="btn btn--secondary w-full"
              >
                <MapPin className="w-4 h-4" />
                {t('camera.viewOnMap')}
              </Link>
            </div>
          </div>

          {/* Detection Stats (for smart cameras) */}
          {camera.type === 'smart' && (
            <div className="card p-6">
              <h3 className="heading-4 mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-primary-500" />
                {t('camera.detectionStats')}
              </h3>
              <div className="space-y-4">
                <InfoItem
                  label={t('camera.totalDetections')}
                  value={detections.length.toString()}
                />
                <InfoItem
                  label={t('camera.lastDetection')}
                  value={
                    detections.length > 0 
                      ? formatDateTime(detections[0].timestamp)
                      : t('common.never')
                  }
                />
                <InfoItem
                  label={t('camera.avgConfidence')}
                  value={
                    detections.length > 0
                      ? `${Math.round(detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length * 100)}%`
                      : '—'
                  }
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Info item component
interface InfoItemProps {
  label: string
  value: React.ReactNode
  mono?: boolean
}

function InfoItem({ label, value, mono }: InfoItemProps) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-sm text-neutral-500 flex-shrink-0">{label}</span>
      <div className={`text-sm text-neutral-900 text-right ${mono ? 'font-mono' : ''}`}>
        {value}
      </div>
    </div>
  )
}
