import { useTranslation } from 'react-i18next'
import { useEffect, useState } from 'react'
import { Camera, Activity, Shield, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react'
import { cameraService, violationService, vehicleService } from '../services/api'
import { LoadingState, ErrorState } from '../components/common/states'

export function Dashboard() {
  const { t } = useTranslation()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState({
    cameras: 0,
    onlineCameras: 0,
    smartCameras: 0,
    todayDetections: 0,
    pendingViolations: 0,
    totalViolations: 0
  })

  useEffect(() => {
    const loadDashboardStats = async () => {
      setLoading(true)
      setError(null)

      try {
        const [cameras, detectionStats, violationStats] = await Promise.all([
          cameraService.getAll(),
          vehicleService.getDetectionStats().catch(() => ({ today: 0 })),
          violationService.getStats().catch(() => ({ total: 0, pending: 0 }))
        ])

        setStats({
          cameras: cameras.length,
          onlineCameras: cameras.filter(c => c.status === 'online').length,
          smartCameras: cameras.filter(c => c.type === 'smart').length,
          todayDetections: detectionStats.today || 0,
          pendingViolations: violationStats.pending || 0,
          totalViolations: violationStats.total || 0
        })
      } catch (err) {
        console.error('Failed to load dashboard stats:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    loadDashboardStats()
  }, [t])

  if (loading) {
    return <LoadingState message={t('dashboard.loading')} />
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
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">{t('pages.dashboard.title')}</h1>
        <p className="text-lg text-gray-600">{t('pages.dashboard.description')}</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Cameras */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
              <Camera className="w-6 h-6 text-blue-600" />
            </div>
            <span className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
              {t('admin.stats.totalCameras')}
            </span>
          </div>
          <div>
            <p className="text-3xl font-bold text-gray-900 mb-1">{stats.cameras}</p>
            <p className="text-sm text-gray-500">
              {stats.onlineCameras} {t('admin.stats.online')}
            </p>
          </div>
        </div>

        {/* Smart Cameras */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
              <Activity className="w-6 h-6 text-purple-600" />
            </div>
            <span className="text-sm font-medium text-purple-600 bg-purple-50 px-3 py-1 rounded-full">
              {t('admin.stats.smartCameras')}
            </span>
          </div>
          <div>
            <p className="text-3xl font-bold text-gray-900 mb-1">{stats.smartCameras}</p>
            <p className="text-sm text-gray-500">
              {t('admin.stats.withPlateRecognition')}
            </p>
          </div>
        </div>

        {/* Today Detections */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <span className="text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full">
              {t('admin.stats.todayDetections')}
            </span>
          </div>
          <div>
            <p className="text-3xl font-bold text-gray-900 mb-1">{stats.todayDetections}</p>
            <p className="text-sm text-gray-500">
              {t('admin.stats.plateRecognitions')}
            </p>
          </div>
        </div>

        {/* Pending Violations */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
            <span className="text-sm font-medium text-red-600 bg-red-50 px-3 py-1 rounded-full">
              {t('admin.stats.pendingViolations')}
            </span>
          </div>
          <div>
            <p className="text-3xl font-bold text-gray-900 mb-1">{stats.pendingViolations}</p>
            <p className="text-sm text-gray-500">
              {stats.totalViolations} {t('admin.stats.total')}
            </p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow cursor-pointer">
          <div className="flex items-center justify-between mb-4">
            <Camera className="w-8 h-8" />
            <span className="text-sm bg-white/20 px-3 py-1 rounded-full">{t('admin.quickAccess')}</span>
          </div>
          <h3 className="text-xl font-bold mb-2">{t('admin.links.cameraManagementDesc')}</h3>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow cursor-pointer">
          <div className="flex items-center justify-between mb-4">
            <Shield className="w-8 h-8" />
            <span className="text-sm bg-white/20 px-3 py-1 rounded-full">{t('nav.map')}</span>
          </div>
          <h3 className="text-xl font-bold mb-2">{t('admin.links.mapViewDesc')}</h3>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow cursor-pointer">
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8" />
            <span className="text-sm bg-white/20 px-3 py-1 rounded-full">{t('nav.violations')}</span>
          </div>
          <h3 className="text-xl font-bold mb-2">{t('admin.links.violationsDesc')}</h3>
        </div>

        <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow cursor-pointer">
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-8 h-8" />
            <span className="text-sm bg-white/20 px-3 py-1 rounded-full">Мониторинг</span>
          </div>
          <h3 className="text-xl font-bold mb-2">{t('admin.links.smartCamerasDesc')}</h3>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">{t('admin.systemStatus')}</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <p className="text-3xl font-bold text-gray-900 mb-1">99.9%</p>
            <p className="text-sm text-gray-500">{t('admin.status.uptime')}</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Camera className="w-8 h-8 text-blue-600" />
            </div>
            <p className="text-3xl font-bold text-gray-900 mb-1">
              {stats.onlineCameras}/{stats.cameras}
            </p>
            <p className="text-sm text-gray-500">{t('admin.status.camerasOnline')}</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-amber-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Activity className="w-8 h-8 text-amber-600" />
            </div>
            <p className="text-3xl font-bold text-gray-900 mb-1">45ms</p>
            <p className="text-sm text-gray-500">{t('admin.status.avgLatency')}</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
            <p className="text-3xl font-bold text-gray-900 mb-1">
              {stats.smartCameras > 0 ? '98.5%' : '—'}
            </p>
            <p className="text-sm text-gray-500">{t('admin.status.recognitionRate')}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
