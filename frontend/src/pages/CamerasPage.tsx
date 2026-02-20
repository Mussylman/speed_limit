/**
 * CamerasPage - Main camera viewing page
 * Requirements: 1.1, 1.2, 1.4
 */

import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { CameraGrid } from '../components/camera'
import { useCameraStore } from '../stores/cameraStore'
import { cameraService } from '../services/api'
import { LoadingState, ErrorState } from '../components/common/states'

export function CamerasPage() {
  const { t } = useTranslation()
  const { cameras, setCameras, viewMode, setViewMode, selectedCameraId, selectCamera } =
    useCameraStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load cameras from API
  useEffect(() => {
    const loadCameras = async () => {
      if (cameras.length > 0) return // Already loaded

      setLoading(true)
      setError(null)

      try {
        const data = await cameraService.getAll()
        setCameras(data)
      } catch (err) {
        console.error('Failed to load cameras:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    loadCameras()
  }, [cameras.length, setCameras, t])

  if (loading) {
    return <LoadingState message={t('cameras.loading')} />
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
    <div className="h-full">
      <CameraGrid
        cameras={cameras}
        viewMode={viewMode}
        selectedCameraId={selectedCameraId}
        onCameraSelect={selectCamera}
        onViewModeChange={setViewMode}
      />
    </div>
  )
}
