/**
 * ViolationFilters Component - Filter controls for violations
 * Requirements: 6.5 - Filter violations by date, type, plate, and status
 *
 * Aesthetic: Clean, functional filter panel with Arctic Command Center styling
 */

import { useTranslation } from 'react-i18next'
import { Calendar, X, Search, RefreshCw } from 'lucide-react'
import type {
  ViolationFilters as ViolationFiltersType,
  ViolationType,
  ViolationStatus,
} from '../../types'

export interface ViolationFiltersProps {
  filters: ViolationFiltersType
  onFilterChange: (filters: Partial<ViolationFiltersType>) => void
  onClearFilters: () => void
}

const VIOLATION_TYPES: ViolationType[] = [
  'speed_limit',
  'red_light',
  'wrong_lane',
  'no_seatbelt',
  'phone_usage',
  'parking',
  'other',
]

const VIOLATION_STATUSES: ViolationStatus[] = ['pending', 'confirmed', 'dismissed']

export function ViolationFilters({
  filters,
  onFilterChange,
  onClearFilters,
}: ViolationFiltersProps) {
  const { t } = useTranslation()

  const handleDateFromChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    onFilterChange({ dateFrom: value ? new Date(value) : undefined })
  }

  const handleDateToChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    onFilterChange({ dateTo: value ? new Date(value) : undefined })
  }

  const handleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value as ViolationType | ''
    onFilterChange({ type: value || undefined })
  }

  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value as ViolationStatus | ''
    onFilterChange({ status: value || undefined })
  }

  const handlePlateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase()
    onFilterChange({ plate: value || undefined })
  }

  const formatDateForInput = (date: Date | undefined): string => {
    if (!date) return ''
    return date.toISOString().split('T')[0]
  }

  const hasActiveFilters = Object.values(filters).some(Boolean)

  return (
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-900 flex items-center gap-3">
          <Search className="w-6 h-6 text-blue-500" />
          {t('violations.filters')}
        </h3>
        {hasActiveFilters && (
          <button
            onClick={onClearFilters}
            className="flex items-center gap-2 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 font-medium rounded-xl transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            {t('violations.clearFilters')}
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        {/* Date From */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <Calendar className="w-4 h-4 text-gray-500" />
            {t('violations.dateFrom')}
          </label>
          <input
            type="date"
            value={formatDateForInput(filters.dateFrom)}
            onChange={handleDateFromChange}
            className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all"
          />
        </div>

        {/* Date To */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <Calendar className="w-4 h-4 text-gray-500" />
            {t('violations.dateTo')}
          </label>
          <input
            type="date"
            value={formatDateForInput(filters.dateTo)}
            onChange={handleDateToChange}
            className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all"
          />
        </div>

        {/* Violation Type */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">{t('violations.type')}</label>
          <select 
            value={filters.type || ''} 
            onChange={handleTypeChange} 
            className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all"
          >
            <option value="">{t('violations.allTypes')}</option>
            {VIOLATION_TYPES.map((type) => (
              <option key={type} value={type}>
                {t(`violation.types.${type}`)}
              </option>
            ))}
          </select>
        </div>

        {/* Status */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">{t('violations.status')}</label>
          <select 
            value={filters.status || ''} 
            onChange={handleStatusChange} 
            className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all"
          >
            <option value="">{t('violations.allStatuses')}</option>
            {VIOLATION_STATUSES.map((status) => (
              <option key={status} value={status}>
                {t(`violation.${status}`)}
              </option>
            ))}
          </select>
        </div>

        {/* Plate Search */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">{t('violations.plateSearch')}</label>
          <div className="relative">
            <input
              type="text"
              value={filters.plate || ''}
              onChange={handlePlateChange}
              placeholder={t('violations.plateSearchPlaceholder')}
              className="w-full pl-12 pr-12 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-500 font-mono uppercase focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all"
            />
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            {filters.plate && (
              <button
                onClick={() => onFilterChange({ plate: undefined })}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
