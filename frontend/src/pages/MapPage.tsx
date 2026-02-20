/**
 * MapPage - Shymkent Camera Map View
 * Requirements: 4.1 (map display), 4.2 (camera markers), 4.3 (click to stream),
 *               4.4 (add camera), 4.5 (edit/delete), 4.6 (update location)
 */

import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { MapView, CameraMarker, RouteVisualizer, MapEditControls } from '../components/map'
import { useCameraStore } from '../stores/cameraStore'
import { useMapStore } from '../stores/mapStore'
import type { Camera } from '../types'
import { Video, MapPin } from 'lucide-react'

export function MapPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()

  const { cameras, selectedCameraId, selectCamera, updateCamera } = useCameraStore()
  const { isEditMode, selectedRoute, setEditMode } = useMapStore()

  // Handle camera marker click - navigate to camera detail
  const handleCameraClick = useCallback(
    (camera: Camera) => {
      selectCamera(camera.id)
      if (!isEditMode) {
        navigate(`/cameras/${camera.id}`)
      }
    },
    [selectCamera, navigate, isEditMode]
  )

  // Handle camera drag end - update camera location
  const handleCameraDragEnd = useCallback(
    (camera: Camera, lat: number, lng: number) => {
      updateCamera(camera.id, {
        location: { ...camera.location, lat, lng },
      })
    },
    [updateCamera]
  )

  // Handle map click in edit mode - set pending location for new camera
  const handleMapClick = useCallback(
    (lat: number, lng: number) => {
      if (isEditMode) {
        // Navigate to camera form with location
        navigate(`/admin/cameras/new?lat=${lat}&lng=${lng}`)
      }
    },
    [isEditMode, navigate]
  )

  // Handle add camera button click
  const handleAddCamera = useCallback(() => {
    // Just enable edit mode - user will click on map
    setEditMode(true)
  }, [setEditMode])

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{t('pages.map.title')}</h1>
            <p className="text-lg text-gray-600 mt-1">{t('pages.map.description')}</p>
          </div>

          {/* Camera count badge */}
          <div className="flex items-center gap-3 px-4 py-2 bg-blue-50 text-blue-700 rounded-xl border border-blue-200">
            <Video size={20} className="text-blue-600" />
            <span className="text-sm font-semibold">
              {cameras.length} {t('nav.cameras').toLowerCase()}
            </span>
          </div>
        </div>
      </div>

      {/* Map container */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden" style={{ height: '700px' }}>
        <MapView className="h-full w-full" onMapClick={handleMapClick}>
          {/* Camera markers */}
          {cameras.map((camera) => (
            <CameraMarker
              key={camera.id}
              camera={camera}
              isSelected={camera.id === selectedCameraId}
              isDraggable={isEditMode}
              onClick={handleCameraClick}
              onDragEnd={handleCameraDragEnd}
            />
          ))}

          {/* Route visualization (if a route is selected) */}
          {selectedRoute && <RouteVisualizer route={selectedRoute} showTimestamps animateRoute />}
        </MapView>

        {/* Edit controls overlay */}
        <MapEditControls onAddCamera={handleAddCamera} />

        {/* Empty state when no cameras */}
        {cameras.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-[1000]">
            <div className="bg-white rounded-2xl p-8 shadow-lg text-center pointer-events-auto border border-gray-100">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-100 flex items-center justify-center">
                <MapPin size={32} className="text-gray-400" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">{t('map.noCameras')}</h3>
              <p className="text-gray-600 mb-6">{t('map.addFirstCamera')}</p>
              <button 
                onClick={handleAddCamera} 
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-semibold rounded-xl transition-all shadow-md hover:shadow-lg transform hover:scale-105"
              >
                {t('map.addCamera')}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
