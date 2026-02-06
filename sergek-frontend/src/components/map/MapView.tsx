/**
 * MapView Component - Leaflet Map for Shymkent
 * Requirements: 4.1 (Shymkent city map display)
 */

import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, useMap, ZoomControl } from 'react-leaflet'
import { useTranslation } from 'react-i18next'
import { useMapStore, SHYMKENT_CENTER, DEFAULT_ZOOM } from '../../stores/mapStore'
import 'leaflet/dist/leaflet.css'

interface MapViewProps {
  children?: React.ReactNode
  className?: string
  onMapClick?: (lat: number, lng: number) => void
}

// Inner component to sync map state with store
function MapStateSync() {
  const map = useMap()
  const { center, zoom, setCenter, setZoom } = useMapStore()

  useEffect(() => {
    map.setView([center.lat, center.lng], zoom)
  }, [map, center.lat, center.lng, zoom])

  useEffect(() => {
    // Force map to invalidate size after mount
    const timer = setTimeout(() => {
      map.invalidateSize()
    }, 100)

    const handleMoveEnd = () => {
      const mapCenter = map.getCenter()
      setCenter({ lat: mapCenter.lat, lng: mapCenter.lng })
    }

    const handleZoomEnd = () => {
      setZoom(map.getZoom())
    }

    map.on('moveend', handleMoveEnd)
    map.on('zoomend', handleZoomEnd)

    return () => {
      clearTimeout(timer)
      map.off('moveend', handleMoveEnd)
      map.off('zoomend', handleZoomEnd)
    }
  }, [map, setCenter, setZoom])

  return null
}

// Click handler component
function MapClickHandler({ onClick }: { onClick?: (lat: number, lng: number) => void }) {
  const map = useMap()
  const { isEditMode } = useMapStore()

  useEffect(() => {
    if (!onClick || !isEditMode) return

    const handleClick = (e: L.LeafletMouseEvent) => {
      onClick(e.latlng.lat, e.latlng.lng)
    }

    map.on('click', handleClick)
    return () => {
      map.off('click', handleClick)
    }
  }, [map, onClick, isEditMode])

  return null
}

export function MapView({ children, className = '', onMapClick }: MapViewProps) {
  const { t } = useTranslation()
  const containerRef = useRef<HTMLDivElement>(null)
  const { isEditMode } = useMapStore()

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full bg-white ${className}`}
      style={{
        cursor: isEditMode ? 'crosshair' : 'grab',
        minHeight: '400px'
      }}
    >
      <MapContainer
        center={[SHYMKENT_CENTER.lat, SHYMKENT_CENTER.lng]}
        zoom={DEFAULT_ZOOM}
        zoomControl={false}
        className="w-full h-full"
        style={{ 
          background: '#ffffff',
          height: '100%',
          width: '100%',
          borderRadius: '16px'
        }}
      >
        {/* Multiple tile layer options for reliability */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
          minZoom={3}
          crossOrigin={true}
          errorTileUrl="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        />

        {/* Zoom controls positioned bottom-right */}
        <ZoomControl position="bottomright" />

        {/* State synchronization */}
        <MapStateSync />

        {/* Click handler for edit mode */}
        <MapClickHandler onClick={onMapClick} />

        {/* Child components (markers, routes, etc.) */}
        {children}
      </MapContainer>

      {/* Edit mode indicator */}
      {isEditMode && (
        <div className="absolute top-4 left-4 z-[1001] px-4 py-2 bg-blue-500 text-white text-sm font-semibold rounded-xl shadow-lg">
          {t('map.editModeActive')}
        </div>
      )}
    </div>
  )
}
