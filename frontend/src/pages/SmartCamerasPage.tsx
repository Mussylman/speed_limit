/**
 * SmartCamerasPage - Display smart cameras with plate recognition
 * Requirements: 2.1, 2.3, 2.5
 */

import { useEffect, useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Cpu, Activity, Eye } from 'lucide-react'
import { useCameraStore } from '../stores/cameraStore'
import { useVehicleStore } from '../stores/vehicleStore'
import { CameraCard } from '../components/camera'
import { PlateDisplay, PlateDetectionList } from '../components/plate'
import { LoadingState, ErrorState, EmptyState } from '../components/common/states'
import { cameraService, vehicleService } from '../services/api'
import type { Camera, PlateDetection } from '../types'

interface SmartCameraWithDetection {
  camera: Camera
  latestDetection: PlateDetection | null
}

export function SmartCamerasPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { cameras, setCameras } = useCameraStore()
  const { detections, setDetections } = useVehicleStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load data from API
  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      setError(null)

      try {
        // Load cameras and detections in parallel
        const [camerasData, detectionsData] = await Promise.all([
          cameraService.getAll(),
          vehicleService.getDetections()
        ])

        setCameras(camerasData)
        setDetections(detectionsData)
      } catch (err) {
        console.error('Failed to load data:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    if (cameras.length === 0 || detections.length === 0) {
      loadData()
    }
  }, [cameras.length, detections.length, setCameras, setDetections, t])

  // Filter to only smart cameras
  const smartCameras = useMemo(() => cameras.filter((c) => c.type === 'smart'), [cameras])

  // Get latest detection for each smart camera
  const smartCamerasWithDetections: SmartCameraWithDetection[] = useMemo(() => {
    return smartCameras.map((camera) => {
      const cameraDetections = detections.filter((d) => d.cameraId === camera.id)
      const latestDetection =
        cameraDetections.length > 0
          ? cameraDetections.reduce((latest, current) =>
              current.timestamp > latest.timestamp ? current : latest
            )
          : null
      return { camera, latestDetection }
    })
  }, [smartCameras, detections])

  // Get the most recent detection across all smart cameras for hero display
  const heroDetection = useMemo(() => {
    const smartCameraIds = new Set(smartCameras.map((c) => c.id))
    const smartDetections = detections.filter((d) => smartCameraIds.has(d.cameraId))
    if (smartDetections.length === 0) return null
    return smartDetections.reduce((latest, current) =>
      current.timestamp > latest.timestamp ? current : latest
    )
  }, [smartCameras, detections])

  const handleCameraClick = (cameraId: string) => {
    navigate(`/cameras/${cameraId}`)
  }

  const handlePlateClick = (plate: string) => {
    navigate(`/vehicles/${encodeURIComponent(plate)}`)
  }

  if (loading) {
    return <LoadingState message={t('smartCameras.loading')} />
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
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="heading-2 mb-2">{t('pages.smartCameras.title')}</h1>
          <p className="body-base max-w-2xl">{t('pages.smartCameras.description')}</p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-neutral-600">
            <Cpu className="w-5 h-5 text-primary-500" />
            <span className="text-sm font-medium">
              {smartCameras.length} {t('camera.smart', 'Smart')}
            </span>
          </div>
          <div className="flex items-center gap-2 text-neutral-600">
            <Activity className="w-5 h-5 text-success-500" />
            <span className="text-sm font-medium">
              {smartCameras.filter((c) => c.status === 'online').length} {t('camera.online')}
            </span>
          </div>
        </div>
      </div>

      {/* Hero Section - Latest Detection */}
      {heroDetection && (
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-neutral-900 via-neutral-800 to-neutral-900 p-8">
          {/* Background pattern */}
          <div
            className="absolute inset-0 opacity-5"
            style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`,
              backgroundSize: '32px 32px',
            }}
          />

          <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between gap-8">
            {/* Left side - Info */}
            <div className="text-center lg:text-left">
              <div className="flex items-center gap-2 text-primary-400 mb-3">
                <Eye className="w-5 h-5" />
                <span className="text-sm font-semibold uppercase tracking-wider">
                  {t('plate.lastDetected', 'Last Detected')}
                </span>
              </div>
              <p className="text-neutral-400 text-sm">
                {new Intl.DateTimeFormat('default', {
                  dateStyle: 'medium',
                  timeStyle: 'medium',
                }).format(heroDetection.timestamp)}
              </p>
              <p className="text-neutral-500 text-xs mt-1">
                {cameras.find((c) => c.id === heroDetection.cameraId)?.name ||
                  heroDetection.cameraId}
              </p>
            </div>

            {/* Center - Hero Plate */}
            <div className="flex-shrink-0">
              <PlateDisplay
                plate={heroDetection.plate}
                size="hero"
                animated
                onClick={() => handlePlateClick(heroDetection.plate)}
              />
            </div>

            {/* Right side - Confidence */}
            <div className="text-center lg:text-right">
              <span className="text-sm text-neutral-400 uppercase tracking-wider">
                {t('plate.confidence', 'Confidence')}
              </span>
              <div className="text-4xl font-bold text-white mt-1">
                {Math.round(heroDetection.confidence * 100)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Smart Cameras Grid */}
      <div>
        <h2 className="heading-4 mb-4">{t('pages.smartCameras.title')}</h2>

        {smartCameras.length === 0 ? (
          <EmptyState
            icon={Cpu}
            title={t('smartCameras.noCameras')}
            description={t('smartCameras.noCamerasDesc')}
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {smartCamerasWithDetections.map(({ camera, latestDetection }) => (
              <div key={camera.id} className="space-y-3">
                {/* Camera Card */}
                <CameraCard
                  camera={camera}
                  showOverlay={true}
                  onClick={() => handleCameraClick(camera.id)}
                />

                {/* Latest Plate for this camera */}
                {latestDetection ? (
                  <div
                    className="flex items-center justify-center p-3 bg-neutral-50 rounded-xl cursor-pointer hover:bg-neutral-100 transition-colors"
                    onClick={() => handlePlateClick(latestDetection.plate)}
                  >
                    <PlateDisplay plate={latestDetection.plate} size="medium" animated />
                  </div>
                ) : (
                  <div className="flex items-center justify-center p-3 bg-neutral-50 rounded-xl">
                    <span className="text-sm text-neutral-400">{t('plate.noDetections')}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Detections List */}
      <div>
        <h2 className="heading-4 mb-4">{t('plate.recentDetections', 'Recent Detections')}</h2>
        <div className="bg-white rounded-xl border border-neutral-200 p-4">
          <PlateDetectionList detections={detections} maxItems={10} showCamera={true} />
        </div>
      </div>
    </div>
  )
}
