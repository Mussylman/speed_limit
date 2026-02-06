/**
 * AdminPage - Clean White Admin Dashboard
 */

import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import {
  Camera,
  Car,
  AlertTriangle,
  Activity,
  TrendingUp,
  Clock,
  MapPin,
  Eye,
  Settings,
  ChevronRight,
  Zap,
} from 'lucide-react'
import { useCameraStore } from '../stores/cameraStore'
import { useViolationStore } from '../stores/violationStore'
import { useVehicleStore } from '../stores/vehicleStore'
import { LoadingState, ErrorState } from '../components/common/states'
import { cameraService, violationService, vehicleService } from '../services/api'

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  trend?: { value: number; isPositive: boolean }
  color: 'blue' | 'green' | 'amber' | 'red' | 'purple'
}

function StatCard({ title, value, subtitle, icon, trend, color }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200',
    green: 'bg-green-50 border-green-200',
    amber: 'bg-amber-50 border-amber-200',
    red: 'bg-red-50 border-red-200',
    purple: 'bg-purple-50 border-purple-200',
  }

  const iconClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    amber: 'bg-amber-100 text-amber-600',
    red: 'bg-red-100 text-red-600',
    purple: 'bg-purple-100 text-purple-600',
  }

  return (
    <div className={`bg-white rounded-2xl p-6 shadow-sm border ${colorClasses[color]} hover:shadow-md transition-shadow`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-2">{title}</p>
          <p className="text-3xl font-bold text-gray-900 mb-1">{value}</p>
          {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
          {trend && (
            <div className="flex items-center gap-1 mt-3">
              <TrendingUp
                className={`w-4 h-4 ${trend.isPositive ? 'text-green-500' : 'text-red-500 rotate-180'}`}
              />
              <span
                className={`text-sm font-medium ${trend.isPositive ? 'text-green-600' : 'text-red-600'}`}
              >
                {trend.isPositive ? '+' : '-'}
                {Math.abs(trend.value)}%
              </span>
              <span className="text-xs text-gray-400">vs last week</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-xl ${iconClasses[color]}`}>{icon}</div>
      </div>
    </div>
  )
}

interface QuickLinkProps {
  to: string
  icon: React.ReactNode
  title: string
  description: string
  color: 'blue' | 'green' | 'purple' | 'amber'
}

function QuickLink({ to, icon, title, description, color }: QuickLinkProps) {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700',
    green: 'from-green-500 to-green-600 hover:from-green-600 hover:to-green-700',
    purple: 'from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700',
    amber: 'from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700',
  }

  return (
    <Link 
      to={to} 
      className={`block bg-gradient-to-r ${colorClasses[color]} rounded-xl p-5 text-white shadow-lg hover:shadow-xl transition-all transform hover:scale-[1.02] group cursor-pointer active:scale-[0.98]`}
      style={{ textDecoration: 'none' }}
    >
      <div className="flex items-center gap-4">
        <div className="p-3 bg-white/20 rounded-lg group-hover:bg-white/30 transition-colors">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-bold text-lg mb-1 text-white">{title}</p>
          <p className="text-sm text-white/90">{description}</p>
        </div>
        <ChevronRight className="w-6 h-6 text-white/70 group-hover:text-white group-hover:translate-x-1 transition-all flex-shrink-0" />
      </div>
    </Link>
  )
}

export function AdminPage() {
  const { t } = useTranslation()
  const { cameras, setCameras } = useCameraStore()
  const { violations, setViolations } = useViolationStore()
  const { detections, setDetections } = useVehicleStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load dashboard data
  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true)
      setError(null)

      try {
        const [camerasData, violationsData, detectionsData] = await Promise.all([
          cameraService.getAll(),
          violationService.getAll(),
          vehicleService.getDetections()
        ])

        setCameras(camerasData)
        setViolations(violationsData)
        setDetections(detectionsData)
      } catch (err) {
        console.error('Failed to load dashboard data:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    if (cameras.length === 0 || violations.length === 0) {
      loadDashboardData()
    }
  }, [cameras.length, violations.length, setCameras, setViolations, setDetections, t])

  // Calculate statistics
  const totalCameras = cameras.length
  const onlineCameras = cameras.filter((c) => c.status === 'online').length
  const smartCameras = cameras.filter((c) => c.type === 'smart').length
  const totalViolations = violations.length
  const pendingViolations = violations.filter((v) => v.status === 'pending').length
  const todayDetections = detections.length

  if (loading) {
    return <LoadingState message={t('admin.loading')} />
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
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">{t('pages.admin.title')}</h1>
            <p className="text-lg text-gray-600">{t('admin.subtitle')}</p>
          </div>
          <div className="flex items-center gap-3 px-4 py-2 bg-green-50 text-green-700 rounded-xl border border-green-200">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
            <span className="text-sm font-semibold">{t('admin.systemOnline')}</span>
          </div>
        </div>
      </div>

      {/* Main Content - Horizontal Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        
        {/* Left Column - Statistics */}
        <div className="xl:col-span-2 space-y-8">
          
          {/* Statistics Grid */}
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
              <Activity className="w-6 h-6 text-blue-500" />
              {t('admin.systemStats')}
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                title={t('admin.stats.totalCameras')}
                value={totalCameras}
                subtitle={`${onlineCameras} ${t('admin.stats.online')}`}
                icon={<Camera className="w-6 h-6" />}
                color="blue"
              />
              <StatCard
                title={t('admin.stats.smartCameras')}
                value={smartCameras}
                subtitle={t('admin.stats.withPlateRecognition')}
                icon={<Zap className="w-6 h-6" />}
                color="purple"
              />
              <StatCard
                title={t('admin.stats.todayDetections')}
                value={todayDetections.toLocaleString()}
                subtitle={t('admin.stats.plateRecognitions')}
                icon={<Eye className="w-6 h-6" />}
                color="amber"
              />
              <StatCard
                title={t('admin.stats.pendingViolations')}
                value={pendingViolations}
                subtitle={`${totalViolations} ${t('admin.stats.total')}`}
                icon={<AlertTriangle className="w-6 h-6" />}
                color="red"
              />
            </div>
          </div>

          {/* System Status */}
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
              <Settings className="w-6 h-6 text-emerald-500" />
              {t('admin.systemStatus')}
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl p-6 border border-emerald-200">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-emerald-500 rounded-xl flex items-center justify-center">
                    <Car className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-emerald-900">
                      {smartCameras > 0 ? '98.5%' : '—'}
                    </p>
                    <p className="text-sm text-emerald-700">{t('admin.status.recognitionRate')}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border border-blue-200">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                    <Activity className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-blue-900">99.9%</p>
                    <p className="text-sm text-blue-700">{t('admin.status.uptime')}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-amber-50 to-amber-100 rounded-xl p-6 border border-amber-200">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-amber-500 rounded-xl flex items-center justify-center">
                    <Clock className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-amber-900">45ms</p>
                    <p className="text-sm text-amber-700">{t('admin.status.avgLatency')}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl p-6 border border-slate-200">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-slate-500 rounded-xl flex items-center justify-center">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-slate-900">
                      {onlineCameras}/{totalCameras}
                    </p>
                    <p className="text-sm text-slate-700">{t('admin.status.camerasOnline')}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Quick Access */}
        <div className="space-y-8">
          
          {/* Quick Access Links */}
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
              <Zap className="w-6 h-6 text-purple-500" />
              {t('admin.quickAccess')}
            </h2>
            
            <div className="grid grid-cols-1 gap-4">
              <QuickLink
                to="/admin/cameras"
                icon={<Camera className="w-6 h-6" />}
                title={t('admin.links.cameraManagement')}
                description={t('admin.links.cameraManagementDesc')}
                color="blue"
              />
              <QuickLink
                to="/map"
                icon={<MapPin className="w-6 h-6" />}
                title={t('admin.links.mapView')}
                description={t('admin.links.mapViewDesc')}
                color="green"
              />
              <QuickLink
                to="/violations"
                icon={<AlertTriangle className="w-6 h-6" />}
                title={t('admin.links.violations')}
                description={t('admin.links.violationsDesc')}
                color="amber"
              />
              <QuickLink
                to="/smart-cameras"
                icon={<Zap className="w-6 h-6" />}
                title={t('admin.links.smartCameras')}
                description={t('admin.links.smartCamerasDesc')}
                color="purple"
              />
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
              <Clock className="w-6 h-6 text-indigo-500" />
              {t('admin.recentActivity')}
            </h2>
            
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-4 bg-green-50 rounded-xl border border-green-200">
                <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
                  <Eye className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-green-900">{t('admin.activity.plateDetected')}</p>
                  <p className="text-sm text-green-700">123ABC45 - {t('camera.camera')} #3</p>
                </div>
                <span className="text-xs text-green-600">{t('admin.activity.timeAgo', '2 мин')}</span>
              </div>
              
              <div className="flex items-center gap-4 p-4 bg-amber-50 rounded-xl border border-amber-200">
                <div className="w-10 h-10 bg-amber-500 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-amber-900">{t('admin.activity.newViolation')}</p>
                  <p className="text-sm text-amber-700">{t('violation.types.speed_limit')}</p>
                </div>
                <span className="text-xs text-amber-600">{t('admin.activity.timeAgo', '5 мин')}</span>
              </div>
              
              <div className="flex items-center gap-4 p-4 bg-blue-50 rounded-xl border border-blue-200">
                <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                  <Camera className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-blue-900">{t('admin.activity.cameraOnline')}</p>
                  <p className="text-sm text-blue-700">{t('camera.camera')} #7 {t('admin.activity.connected')}</p>
                </div>
                <span className="text-xs text-blue-600">{t('admin.activity.timeAgo', '10 мин')}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
