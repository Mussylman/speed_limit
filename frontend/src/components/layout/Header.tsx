/**
 * Header Component - Clean White Professional Interface
 * Beautiful white header with shadows and clear typography
 */

import { Bell, User, Menu, Wifi, Shield, Activity } from 'lucide-react'
import { LanguageSwitcher } from '../common/LanguageSwitcher'

interface HeaderProps {
  title?: string
  onMenuClick?: () => void
}

export function Header({ title, onMenuClick }: HeaderProps) {

  return (
    <header className="h-18 bg-white border-b border-gray-100 flex items-center justify-between px-6 lg:px-8 shadow-sm relative z-10">
      {/* Left: Menu button (mobile) + Title */}
      <div className="flex items-center gap-6">
        {/* Mobile menu button */}
        <button
          onClick={onMenuClick}
          className="p-3 -ml-3 rounded-xl text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-all lg:hidden"
          aria-label="Open menu"
        >
          <Menu size={24} />
        </button>

        {title && (
          <div className="flex items-center gap-4">
            <div className="w-1.5 h-10 bg-gradient-to-b from-blue-500 to-blue-600 rounded-full shadow-sm" />
            <h1 className="text-2xl lg:text-3xl font-bold text-gray-900 truncate">
              {title}
            </h1>
          </div>
        )}
      </div>

      {/* Center: System Status */}
      <div className="hidden md:flex items-center gap-4">
        <div className="flex items-center gap-2 px-4 py-2 bg-green-50 rounded-xl border border-green-200 shadow-sm">
          <Activity size={16} className="text-green-600" />
          <span className="text-sm font-semibold text-green-700">Онлайн</span>
        </div>
        
        <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-xl border border-blue-200 shadow-sm">
          <Wifi size={16} className="text-blue-600" />
          <span className="text-sm font-semibold text-blue-700">Подключено</span>
        </div>
        
        <div className="flex items-center gap-2 px-4 py-2 bg-amber-50 rounded-xl border border-amber-200 shadow-sm">
          <Shield size={16} className="text-amber-600" />
          <span className="text-sm font-semibold text-amber-700">Защищено</span>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-3">
        {/* Language Switcher */}
        <LanguageSwitcher />

        {/* Notifications */}
        <button className="relative p-3 rounded-xl text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-all shadow-sm hover:shadow-md">
          <Bell size={22} />
          <span className="absolute top-2 right-2 w-2.5 h-2.5 bg-red-500 rounded-full shadow-sm" />
        </button>

        {/* User Profile */}
        <button className="flex items-center gap-3 p-2 rounded-xl text-gray-600 hover:bg-gray-100 transition-all shadow-sm hover:shadow-md">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-md">
            <User size={20} className="text-white" />
          </div>
          <div className="hidden sm:block text-left">
            <div className="text-sm font-semibold text-gray-900">Администратор</div>
            <div className="text-xs text-gray-500">Системный администратор</div>
          </div>
        </button>
      </div>
    </header>
  )
}
