import { useState, useMemo, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import {
  Camera,
  Plus,
  Pencil,
  Trash2,
  Search,
  MapPin,
  Zap,
  ChevronLeft,
  AlertCircle,
  MoreVertical,
  Loader2,
} from 'lucide-react'
import { useCameraStore } from '../stores/cameraStore'
import { cameraService } from '../services/api'
import type { Camera as CameraType, CameraStatus } from '../types'

function StatusBadge({ status }: { status: CameraStatus }) {
  const { t } = useTranslation()
  
  const statusConfig = {
    online: {
      label: t('camera.online'),
      className: 'px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium',
    },
    offline: {
      label: t('camera.offline'),
      className: 'px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm font-medium',
    },
    error: {
      label: t('camera.error'),
      className: 'px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium',
    },
  }

  const config = statusConfig[status]

  return (
    <span className={config.className}>
      {config.label}
    </span>
  )
}

function TypeBadge({ type }: { type: 'smart' | 'standard' }) {
  const { t } = useTranslation()
  
  if (type === 'smart') {
    return (
      <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium flex items-center gap-1">
        <Zap className="w-3 h-3" />
        {t('camera.smart')}
      </span>
    )
  }

  return (
    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium flex items-center gap-1">
      <Camera className="w-3 h-3" />
      {t('camera.standard')}
    </span>
  )
}

interface CameraRowProps {
  camera: CameraType
  onEdit: (id: string) => void
  onDelete: (id: string) => void
}

function CameraRow({ camera, onEdit, onDelete }: CameraRowProps) {
  const [showActions, setShowActions] = useState(false)
  const { t } = useTranslation()

  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
      <td className="py-4 px-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center">
            {camera.type === 'smart' ? (
              <Zap className="w-5 h-5 text-purple-600" />
            ) : (
              <Camera className="w-5 h-5 text-blue-600" />
            )}
          </div>
          <div>
            <p className="font-semibold text-gray-900">{camera.name}</p>
            <p className="text-xs text-gray-500 font-mono">{camera.id}</p>
          </div>
        </div>
      </td>
      <td className="py-4 px-6">
        <TypeBadge type={camera.type} />
      </td>
      <td className="py-4 px-6">
        <StatusBadge status={camera.status} />
      </td>
      <td className="py-4 px-6">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <MapPin className="w-4 h-4 text-gray-400" />
          <span>
            {camera.location.address ||
              `${camera.location.lat.toFixed(4)}, ${camera.location.lng.toFixed(4)}`}
          </span>
        </div>
      </td>
      <td className="py-4 px-6">
        <div className="relative">
          <button
            onClick={() => setShowActions(!showActions)}
            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
          >
            <MoreVertical className="w-4 h-4" />
          </button>
          {showActions && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowActions(false)} />
              <div className="absolute right-0 top-full mt-1 bg-white rounded-xl shadow-lg border border-gray-200 py-2 z-20 min-w-[140px]">
                <button
                  onClick={() => {
                    onEdit(camera.id)
                    setShowActions(false)
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                >
                  <Pencil className="w-4 h-4" />
                  {t('common.edit')}
                </button>
                <button
                  onClick={() => {
                    onDelete(camera.id)
                    setShowActions(false)
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  {t('common.delete')}
                </button>
              </div>
            </>
          )}
        </div>
      </td>
    </tr>
  )
}

export function AdminCamerasPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { cameras, setCameras, removeCamera } = useCameraStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isDeleting, setIsDeleting] = useState(false)

  // Load cameras on mount
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const camerasData = await cameraService.getAll()
        setCameras(camerasData)
      } catch (error) {
        console.error('Failed to load cameras:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadCameras()
  }, [setCameras])

  // Filter cameras based on search
  const filteredCameras = useMemo(() => {
    if (!searchQuery.trim()) return cameras
    const query = searchQuery.toLowerCase()
    return cameras.filter(
      (cam) =>
        cam.name.toLowerCase().includes(query) ||
        cam.id.toLowerCase().includes(query) ||
        cam.location.address?.toLowerCase().includes(query)
    )
  }, [cameras, searchQuery])

  const handleEdit = (id: string) => {
    navigate(`/admin/cameras/${id}/edit`)
  }

  const handleDelete = (id: string) => {
    setDeleteConfirm(id)
  }

  const confirmDelete = async () => {
    if (deleteConfirm) {
      setIsDeleting(true)
      try {
        await cameraService.delete(deleteConfirm)
        removeCamera(deleteConfirm)
        setDeleteConfirm(null)
      } catch (error) {
        console.error('Failed to delete camera:', error)
      } finally {
        setIsDeleting(false)
      }
    }
  }

  // Stats
  const totalCameras = cameras.length
  const onlineCameras = cameras.filter((c) => c.status === 'online').length
  const smartCameras = cameras.filter((c) => c.type === 'smart').length

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
        <div className="flex items-center gap-6">
          <Link to="/admin" className="p-3 rounded-xl bg-gray-100 hover:bg-gray-200 transition-colors">
            <ChevronLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-gray-900">{t('pages.adminCameras.title')}</h1>
            <p className="text-lg text-gray-600 mt-1">{t('pages.adminCameras.description')}</p>
          </div>
          <Link 
            to="/admin/cameras/new" 
            className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-semibold rounded-xl transition-all shadow-md hover:shadow-lg transform hover:scale-105"
          >
            <Plus className="w-5 h-5" />
            {t('admin.cameras.addNew')}
          </Link>
        </div>
      </div>

      {isLoading ? (
        <div className="bg-white rounded-2xl p-16 shadow-sm border border-gray-100">
          <div className="flex items-center justify-center">
            <div className="flex items-center gap-3">
              <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
              <span className="text-lg text-gray-600">{t('common.loading')}</span>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Stats Bar */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                  <Camera className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{totalCameras}</p>
                  <p className="text-sm text-gray-600">{t('admin.stats.totalCameras')}</p>
                </div>
              </div>
              <div className="w-px h-12 bg-gray-200" />
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{onlineCameras}</p>
                  <p className="text-sm text-gray-600">{t('admin.stats.online')}</p>
                </div>
              </div>
              <div className="w-px h-12 bg-gray-200" />
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                  <Zap className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{smartCameras}</p>
                  <p className="text-sm text-gray-600">{t('admin.stats.smartCameras')}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Search */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <div className="relative max-w-md">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={t('common.search')}
                className="w-full pl-12 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all"
              />
            </div>
          </div>

          {/* Table */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="text-left py-4 px-6 text-sm font-semibold text-gray-700">
                    {t('admin.cameras.table.name')}
                  </th>
                  <th className="text-left py-4 px-6 text-sm font-semibold text-gray-700">
                    {t('admin.cameras.table.type')}
                  </th>
                  <th className="text-left py-4 px-6 text-sm font-semibold text-gray-700">
                    {t('admin.cameras.table.status')}
                  </th>
                  <th className="text-left py-4 px-6 text-sm font-semibold text-gray-700">
                    {t('admin.cameras.table.location')}
                  </th>
                  <th className="text-left py-4 px-6 text-sm font-semibold text-gray-700 w-20">
                    {t('admin.cameras.table.actions')}
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredCameras.length > 0 ? (
                  filteredCameras.map((camera) => (
                    <CameraRow
                      key={camera.id}
                      camera={camera}
                      onEdit={handleEdit}
                      onDelete={handleDelete}
                    />
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="py-16 text-center">
                      <div className="flex flex-col items-center gap-4">
                        <div className="w-16 h-16 rounded-2xl bg-gray-100 flex items-center justify-center">
                          <Camera className="w-8 h-8 text-gray-400" />
                        </div>
                        <div>
                          <p className="text-lg font-medium text-gray-900">{t('admin.cameras.noResults')}</p>
                          <p className="text-gray-500 mt-1">{t('violations.noViolationsHint')}</p>
                        </div>
                        {searchQuery && (
                          <button
                            onClick={() => setSearchQuery('')}
                            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                          >
                            {t('violations.clearFilters')}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setDeleteConfirm(null)}
          />
          <div className="relative bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full mx-4">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-red-100 flex items-center justify-center">
                <AlertCircle className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">{t('admin.cameras.confirmDelete')}</h3>
                <p className="text-gray-600 mt-1">{t('admin.cameras.deleteWarning')}</p>
              </div>
            </div>
            <div className="flex justify-end gap-3">
              <button 
                onClick={() => setDeleteConfirm(null)} 
                className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-xl transition-colors"
                disabled={isDeleting}
              >
                {t('common.cancel')}
              </button>
              <button 
                onClick={confirmDelete} 
                className="px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-medium rounded-xl transition-all flex items-center gap-2"
                disabled={isDeleting}
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {t('common.saving')}
                  </>
                ) : (
                  t('common.delete')
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
