/**
 * Sidebar Component - Clean White Professional Interface
 * Beautiful white sidebar with shadows and clear typography
 */

import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  Camera,
  Cpu,
  Map,
  AlertTriangle,
  Settings,
  ChevronLeft,
  ChevronRight,
  X,
  Shield,
} from 'lucide-react'
import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'

interface NavItem {
  path: string
  labelKey: string
  icon: React.ReactNode
  description: string
}

const navItems: NavItem[] = [
  { 
    path: '/', 
    labelKey: 'nav.dashboard', 
    icon: <LayoutDashboard size={24} />, 
    description: 'Общий обзор' 
  },
  { 
    path: '/cameras', 
    labelKey: 'nav.cameras', 
    icon: <Camera size={24} />, 
    description: 'Управление камерами' 
  },
  { 
    path: '/smart-cameras', 
    labelKey: 'nav.smartCameras', 
    icon: <Cpu size={24} />, 
    description: 'Умные камеры' 
  },
  { 
    path: '/map', 
    labelKey: 'nav.map', 
    icon: <Map size={24} />, 
    description: 'Просмотр карты' 
  },
  { 
    path: '/violations', 
    labelKey: 'nav.violations', 
    icon: <AlertTriangle size={24} />, 
    description: 'Записи нарушений' 
  },
  { 
    path: '/admin', 
    labelKey: 'nav.admin', 
    icon: <Settings size={24} />, 
    description: 'Управление системой' 
  },
]

interface SidebarProps {
  isMobileOpen?: boolean
  onMobileClose?: () => void
}

export function Sidebar({ isMobileOpen = false, onMobileClose }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const { t } = useTranslation()

  // Check for mobile viewport
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Close mobile menu on navigation
  const handleNavClick = () => {
    if (isMobile && onMobileClose) {
      onMobileClose()
    }
  }

  const sidebarContent = (
    <>
      {/* Logo Header */}
      <div className="h-20 flex items-center justify-between px-6 border-b border-gray-100 bg-white">
        <div className="flex items-center gap-4 overflow-hidden">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg">
            <Shield className="text-white" size={24} />
          </div>
          {(!collapsed || isMobile) && (
            <div className="flex flex-col">
              <span className="font-bold text-2xl text-gray-900 tracking-tight">SERGEK</span>
              <span className="text-sm text-gray-600 font-medium">Система видеонаблюдения Шымкент</span>
            </div>
          )}
        </div>

        {/* Mobile close button */}
        {isMobile && onMobileClose && (
          <button
            onClick={onMobileClose}
            className="p-3 rounded-xl text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-all lg:hidden"
          >
            <X size={24} />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-8 px-4 overflow-y-auto bg-white">
        <ul className="space-y-3">
          {navItems.map((item, index) => (
            <motion.li 
              key={item.path}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <NavLink
                to={item.path}
                onClick={handleNavClick}
                className={({ isActive }) =>
                  `flex items-center gap-4 px-6 py-5 rounded-2xl transition-all duration-200 group relative ${
                    isActive
                      ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/25'
                      : 'text-gray-700 hover:bg-gray-50 hover:shadow-md'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    {/* Icon */}
                    <span className={`flex-shrink-0 ${isActive ? 'text-white' : 'text-gray-500 group-hover:text-blue-500'}`}>
                      {item.icon}
                    </span>
                    
                    {/* Text Content */}
                    {(!collapsed || isMobile) && (
                      <div className="flex flex-col min-w-0 flex-1">
                        <span className={`font-semibold text-lg truncate ${isActive ? 'text-white' : 'text-gray-900 group-hover:text-blue-600'}`}>
                          {t(item.labelKey)}
                        </span>
                        <span className={`text-sm truncate ${isActive ? 'text-blue-100' : 'text-gray-500 group-hover:text-blue-500'}`}>
                          {item.description}
                        </span>
                      </div>
                    )}
                    
                    {/* Active indicator */}
                    {isActive && (
                      <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-white rounded-l-full" />
                    )}
                  </>
                )}
              </NavLink>
            </motion.li>
          ))}
        </ul>
      </nav>

      {/* System Status */}
      <div className="p-6 border-t border-gray-100 bg-white">
        <div className="bg-gray-50 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse shadow-sm" />
            <span className="text-sm font-semibold text-green-600">Система онлайн</span>
          </div>
          <div className="text-xs text-gray-600 space-y-1">
            <div>Время работы: 99.9%</div>
            <div>Активные камеры: 24/7</div>
          </div>
        </div>
      </div>

      {/* Collapse Toggle - Desktop only */}
      {!isMobile && (
        <div className="p-4 border-t border-gray-100 bg-white">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 rounded-xl text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-all shadow-sm hover:shadow-md"
          >
            <div className={`transform transition-transform ${collapsed ? 'rotate-180' : ''}`}>
              {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            </div>
            {!collapsed && <span className="text-sm font-medium">{t('nav.collapse')}</span>}
          </button>
        </div>
      )}
    </>
  )

  // Mobile: Drawer overlay
  if (isMobile) {
    return (
      <AnimatePresence>
        {isMobileOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
              onClick={onMobileClose}
            />

            {/* Drawer */}
            <motion.aside
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
              className="fixed left-0 top-0 h-screen w-80 bg-white flex flex-col z-50 lg:hidden shadow-2xl"
            >
              {sidebarContent}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    )
  }

  // Desktop: Static sidebar
  return (
    <aside
      className="hidden lg:flex h-screen bg-white flex-col transition-all duration-300 ease-out shadow-xl border-r border-gray-100"
      style={{ width: collapsed ? '80px' : '320px' }}
    >
      {sidebarContent}
    </aside>
  )
}
