/**
 * CameraFormPage - Professional Camera Management Form
 * Elegant, horizontal layout with refined aesthetics
 */

import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate, Link, useSearchParams } from 'react-router-dom'
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet'
import { useTranslation } from 'react-i18next'
import L from 'leaflet'
import { 
  ChevronLeft, 
  Camera as CameraIcon, 
  MapPin, 
  Save, 
  X, 
  AlertCircle, 
  Loader2,
  CheckCircle,
  Wifi,
  Globe
} from 'lucide-react'
import { useCameraStore } from '../stores/cameraStore'
import { cameraService } from '../services/api'
import { LanguageSwitcher } from '../components/common'
import type { Camera, CameraType, GeoLocation } from '../types'

// Shymkent merkez koordinatları
const SHYMKENT_CENTER: [number, number] = [42.3417, 69.5901]

// Custom marker icon for location picker
const locationIcon = new L.Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="40" viewBox="0 0 32 40">
      <path d="M16 0C7.163 0 0 7.163 0 16c0 12 16 24 16 24s16-12 16-24C32 7.163 24.837 0 16 0z" fill="#3b82f6"/>
      <circle cx="16" cy="16" r="8" fill="white"/>
      <circle cx="16" cy="16" r="4" fill="#3b82f6"/>
    </svg>
  `),
  iconSize: [32, 40],
  iconAnchor: [16, 40],
  popupAnchor: [0, -40],
})

interface FormData {
  name: string
  rtspUrl: string
  type: CameraType
  location: GeoLocation | null
  lane?: number
}

interface FormErrors {
  name?: string
  rtspUrl?: string
  location?: string
}

// Harita tıklama handler bileşeni
function LocationPicker({
  onLocationSelect,
  selectedLocation,
}: {
  onLocationSelect: (location: GeoLocation) => void
  selectedLocation: GeoLocation | null
}) {
  useMapEvents({
    click: (e) => {
      onLocationSelect({
        lat: e.latlng.lat,
        lng: e.latlng.lng,
      })
    },
  })

  if (!selectedLocation) return null

  return (
    <Marker
      position={[selectedLocation.lat, selectedLocation.lng]}
      icon={locationIcon}
      draggable
      eventHandlers={{
        dragend: (e) => {
          const marker = e.target
          const position = marker.getLatLng()
          onLocationSelect({
            lat: position.lat,
            lng: position.lng,
            address: selectedLocation.address,
          })
        },
      }}
    />
  )
}

export function CameraFormPage() {
  const { t } = useTranslation()
  const { id } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { addCamera, updateCamera, getCameraById } = useCameraStore()
  const isEdit = Boolean(id)

  const [formData, setFormData] = useState<FormData>({
    name: '',
    rtspUrl: '',
    type: 'smart', // Default to smart camera
    location: null,
    lane: 1,
  })
  const [errors, setErrors] = useState<FormErrors>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle')

  // Load camera data in edit mode
  useEffect(() => {
    const loadCamera = async () => {
      if (isEdit && id) {
        setIsLoading(true)
        try {
          let camera = getCameraById(id)
          if (!camera) {
            camera = await cameraService.getById(id)
          }
          
          if (camera) {
            setFormData({
              name: camera.name,
              rtspUrl: camera.rtspUrl,
              type: camera.type,
              location: camera.location,
              lane: camera.lane || 1,
            })
          }
        } catch (err) {
          console.error('Failed to load camera:', err)
        } finally {
          setIsLoading(false)
        }
      }
    }

    loadCamera()
  }, [isEdit, id, getCameraById])

  // Set initial location from URL parameters (when coming from map)
  useEffect(() => {
    const lat = searchParams.get('lat')
    const lng = searchParams.get('lng')
    
    if (lat && lng && !formData.location) {
      setFormData(prev => ({
        ...prev,
        location: {
          lat: parseFloat(lat),
          lng: parseFloat(lng)
        }
      }))
    }
  }, [searchParams, formData.location])

  const validateForm = useCallback((): boolean => {
    const newErrors: FormErrors = {}

    if (!formData.name.trim()) {
      newErrors.name = t('admin.form.validation.nameRequired')
    }

    if (!formData.rtspUrl.trim()) {
      newErrors.rtspUrl = t('admin.form.validation.rtspRequired')
    } else if (!formData.rtspUrl.startsWith('rtsp://')) {
      newErrors.rtspUrl = t('admin.form.validation.rtspFormat')
    }

    if (!formData.location) {
      newErrors.location = t('admin.form.validation.locationRequired')
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }, [formData, t])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) return

    setIsSubmitting(true)

    try {
      const cameraData: Omit<Camera, 'id' | 'status' | 'hlsUrl'> = {
        name: formData.name.trim(),
        rtspUrl: formData.rtspUrl.trim(),
        type: formData.type,
        location: formData.location!,
        lane: formData.lane,
      }

      if (isEdit && id) {
        const updatedCamera = await cameraService.update(id, cameraData)
        updateCamera(id, updatedCamera)
      } else {
        const newCamera = await cameraService.create(cameraData)
        addCamera(newCamera)
      }

      navigate('/admin/cameras')
    } catch (error) {
      console.error('Failed to save camera:', error)
      setErrors({ name: t('admin.form.validation.saveError') })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleLocationSelect = (location: GeoLocation) => {
    setFormData((prev) => ({ ...prev, location }))
    if (errors.location) {
      setErrors((prev) => ({ ...prev, location: undefined }))
    }
  }

  const handleTestConnection = async () => {
    if (!formData.rtspUrl.trim()) return

    setConnectionStatus('testing')
    try {
      const result = await cameraService.testConnection(formData.rtspUrl)
      if (result.success) {
        setConnectionStatus('success')
        setErrors(prev => ({ ...prev, rtspUrl: undefined }))
      } else {
        setConnectionStatus('error')
        setErrors(prev => ({ ...prev, rtspUrl: result.message }))
      }
    } catch (err) {
      setConnectionStatus('error')
      setErrors(prev => ({ ...prev, rtspUrl: t('admin.form.connectionTestFailed') }))
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-4 bg-white px-8 py-6 rounded-2xl shadow-lg border border-gray-100">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="text-xl font-medium text-gray-700">{t('common.loading')}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header with Language Switcher */}
        <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100 mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link 
                to="/admin/cameras" 
                className="flex items-center justify-center w-14 h-14 rounded-2xl bg-gray-100 hover:bg-gray-200 transition-all duration-300 hover:scale-105 border border-gray-200 group"
              >
                <ChevronLeft className="w-6 h-6 text-gray-600 group-hover:text-blue-600 transition-colors" />
              </Link>
              <div>
                <h1 className="text-4xl font-bold text-gray-900">
                  {isEdit ? t('pages.cameraForm.editTitle') : t('pages.cameraForm.createTitle')}
                </h1>
                <p className="text-xl text-gray-600 mt-2">
                  {t('pages.cameraForm.description')}
                </p>
              </div>
            </div>
            
            {/* Language Switcher */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-gray-500">
                <Globe className="w-5 h-5" />
                <span className="text-sm font-medium">{t('common.language')}</span>
              </div>
              <LanguageSwitcher />
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Main Content - Horizontal Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Left Column - Form Fields */}
            <div className="space-y-8">
              
              {/* Camera Information Card */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="bg-gradient-to-r from-slate-600 to-slate-700 px-8 py-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-4">
                    <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                      <CameraIcon className="w-6 h-6 text-white" />
                    </div>
                    {t('admin.form.basicInfo')}
                  </h2>
                </div>
                
                <div className="p-8 space-y-8">
                  {/* Camera Name */}
                  <div className="space-y-4">
                    <label className="block text-lg font-semibold text-gray-800">
                      {t('admin.form.name')} <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => {
                        setFormData((prev) => ({ ...prev, name: e.target.value }))
                        if (errors.name) setErrors((prev) => ({ ...prev, name: undefined }))
                      }}
                      placeholder={t('admin.form.namePlaceholder')}
                      className={`w-full px-6 py-4 text-lg rounded-xl border-2 transition-all duration-300 focus:outline-none focus:ring-4 ${
                        errors.name 
                          ? 'border-red-300 bg-red-50 focus:border-red-500 focus:ring-red-100' 
                          : 'border-gray-200 bg-gray-50 focus:border-slate-500 focus:ring-slate-100 hover:border-gray-300'
                      }`}
                    />
                    {errors.name && (
                      <p className="text-red-600 flex items-center gap-2 text-base font-medium">
                        <AlertCircle className="w-5 h-5" />
                        {errors.name}
                      </p>
                    )}
                  </div>

                  {/* RTSP URL */}
                  <div className="space-y-4">
                    <label className="block text-lg font-semibold text-gray-800">
                      {t('admin.form.rtspUrl')} <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      value={formData.rtspUrl}
                      onChange={(e) => {
                        setFormData((prev) => ({ ...prev, rtspUrl: e.target.value }))
                        if (errors.rtspUrl) setErrors((prev) => ({ ...prev, rtspUrl: undefined }))
                        setConnectionStatus('idle')
                      }}
                      placeholder={t('admin.form.rtspUrlPlaceholder')}
                      className={`w-full px-6 py-4 text-lg font-mono rounded-xl border-2 transition-all duration-300 focus:outline-none focus:ring-4 ${
                        errors.rtspUrl 
                          ? 'border-red-300 bg-red-50 focus:border-red-500 focus:ring-red-100' 
                          : 'border-gray-200 bg-gray-50 focus:border-slate-500 focus:ring-slate-100 hover:border-gray-300'
                      }`}
                    />
                    {errors.rtspUrl && (
                      <p className="text-red-600 flex items-center gap-2 text-base font-medium">
                        <AlertCircle className="w-5 h-5" />
                        {errors.rtspUrl}
                      </p>
                    )}
                  </div>

                  {/* Lane Selection */}
                  <div className="space-y-4">
                    <label className="block text-lg font-semibold text-gray-800">
                      {t('admin.form.lane')}
                    </label>
                    <select
                      value={formData.lane}
                      onChange={(e) => setFormData((prev) => ({ ...prev, lane: parseInt(e.target.value) }))}
                      className="w-full px-6 py-4 text-lg rounded-xl border-2 border-gray-200 bg-gray-50 focus:border-slate-500 focus:ring-4 focus:ring-slate-100 focus:outline-none transition-all duration-300"
                    >
                      <option value={1}>{t('admin.form.lane')} 1</option>
                      <option value={2}>{t('admin.form.lane')} 2</option>
                      <option value={3}>{t('admin.form.lane')} 3</option>
                      <option value={4}>{t('admin.form.lane')} 4</option>
                    </select>
                  </div>

                  {/* Address (optional) */}
                  <div className="space-y-4">
                    <label className="block text-lg font-semibold text-gray-800">
                      {t('admin.form.address')}
                    </label>
                    <input
                      type="text"
                      value={formData.location?.address || ''}
                      onChange={(e) => {
                        if (formData.location) {
                          setFormData((prev) => ({
                            ...prev,
                            location: { ...prev.location!, address: e.target.value },
                          }))
                        }
                      }}
                      placeholder={t('admin.form.addressPlaceholder')}
                      disabled={!formData.location}
                      className="w-full px-6 py-4 text-lg rounded-xl border-2 border-gray-200 bg-gray-50 focus:border-slate-500 focus:ring-4 focus:ring-slate-100 focus:outline-none transition-all duration-300 disabled:bg-gray-100 disabled:text-gray-400"
                    />
                  </div>
                </div>
              </div>

              {/* Connection Test Button */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
                <h3 className="text-lg font-semibold text-gray-800 mb-6">{t('admin.form.testConnection')}</h3>
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={!formData.rtspUrl.trim() || connectionStatus === 'testing'}
                  className={`w-full px-8 py-5 font-semibold rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none text-lg flex items-center justify-center gap-4 ${
                    connectionStatus === 'success' 
                      ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white'
                      : connectionStatus === 'error'
                      ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white'
                      : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed'
                  }`}
                >
                  {connectionStatus === 'testing' ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      {t('admin.form.testing')}
                    </>
                  ) : connectionStatus === 'success' ? (
                    <>
                      <CheckCircle className="w-6 h-6" />
                      {t('admin.form.connectionSuccess')}
                    </>
                  ) : connectionStatus === 'error' ? (
                    <>
                      <X className="w-6 h-6" />
                      {t('admin.form.connectionError')}
                    </>
                  ) : (
                    <>
                      <Wifi className="w-6 h-6" />
                      {t('admin.form.testConnection')}
                    </>
                  )}
                </button>
              </div>

              {/* Save Button */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
                <h3 className="text-lg font-semibold text-gray-800 mb-6">{t('common.save')}</h3>
                <button 
                  type="submit" 
                  disabled={isSubmitting} 
                  className="w-full px-8 py-5 bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white font-semibold rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed disabled:transform-none text-lg flex items-center justify-center gap-4"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      {t('common.saving')}
                    </>
                  ) : (
                    <>
                      <Save className="w-6 h-6" />
                      {t('common.save')}
                    </>
                  )}
                </button>
              </div>

              {/* Cancel Button */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
                <h3 className="text-lg font-semibold text-gray-800 mb-6">{t('common.cancel')}</h3>
                <Link 
                  to="/admin/cameras" 
                  className="w-full px-8 py-5 bg-white hover:bg-gray-50 text-gray-700 font-semibold rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-[1.02] border-2 border-gray-200 hover:border-gray-300 text-lg flex items-center justify-center gap-4"
                >
                  <X className="w-6 h-6" />
                  {t('common.cancel')}
                </Link>
              </div>
            </div>

            {/* Right Column - Map */}
            <div className="space-y-6">
              
              {/* Location Card */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden h-full">
                <div className="bg-gradient-to-r from-emerald-500 to-emerald-600 px-8 py-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-4">
                    <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                      <MapPin className="w-6 h-6 text-white" />
                    </div>
                    {t('admin.form.location')}
                  </h2>
                </div>

                <div className="p-8 space-y-6 h-full">
                  <div className="bg-emerald-50 border-2 border-emerald-200 rounded-xl p-4">
                    <p className="text-emerald-800 font-semibold text-base">📍 {t('admin.form.selectLocation')}</p>
                  </div>

                  {/* Map */}
                  <div
                    className={`flex-1 rounded-xl overflow-hidden border-2 shadow-lg ${
                      errors.location ? 'border-red-300' : 'border-gray-200'
                    }`}
                    style={{ minHeight: '400px' }}
                  >
                    <MapContainer
                      center={
                        formData.location ? [formData.location.lat, formData.location.lng] : SHYMKENT_CENTER
                      }
                      zoom={13}
                      className="h-full w-full"
                      style={{ cursor: 'crosshair', minHeight: '400px' }}
                    >
                      <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                      />
                      <LocationPicker
                        onLocationSelect={handleLocationSelect}
                        selectedLocation={formData.location}
                      />
                    </MapContainer>
                  </div>

                  {errors.location && (
                    <div className="bg-red-50 border-2 border-red-200 rounded-xl p-4">
                      <p className="text-red-700 flex items-center gap-2 text-base font-medium">
                        <AlertCircle className="w-5 h-5" />
                        {errors.location}
                      </p>
                    </div>
                  )}

                  {/* Coordinates Display */}
                  {formData.location && (
                    <div className="bg-gray-50 rounded-xl p-6">
                      <h3 className="text-lg font-semibold text-gray-800 mb-4">📍 {t('admin.form.selectedLocation')}</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <label className="block text-sm font-medium text-gray-600">
                            {t('admin.form.latitude')}
                          </label>
                          <input
                            type="text"
                            value={formData.location.lat.toFixed(6)}
                            readOnly
                            className="w-full px-4 py-3 bg-white border-2 border-gray-200 rounded-lg font-mono text-sm text-gray-700"
                          />
                        </div>
                        <div className="space-y-2">
                          <label className="block text-sm font-medium text-gray-600">
                            {t('admin.form.longitude')}
                          </label>
                          <input
                            type="text"
                            value={formData.location.lng.toFixed(6)}
                            readOnly
                            className="w-full px-4 py-3 bg-white border-2 border-gray-200 rounded-lg font-mono text-sm text-gray-700"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}