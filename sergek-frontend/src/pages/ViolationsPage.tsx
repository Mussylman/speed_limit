/**
 * ViolationsPage - Traffic violations listing page
 * Requirements: 6.1 - Display all traffic violations with pagination and sorting
 */

import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Filter,
} from 'lucide-react'
import { useViolationStore } from '../stores/violationStore'
import { ViolationFilters } from '../components/violation/ViolationFilters'
import { ViolationCard } from '../components/violation/ViolationCard'
import { LoadingState, ErrorState } from '../components/common/states'
import { violationService } from '../services/api'
import type { ViolationStatus } from '../types'

type SortField = 'timestamp' | 'type' | 'status'
type SortDirection = 'asc' | 'desc'

const ITEMS_PER_PAGE = 6

export function ViolationsPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const {
    violations,
    setViolations,
    filters,
    setFilters,
    clearFilters,
    getFilteredViolations,
    updateViolationStatus,
  } = useViolationStore()

  const [sortField, setSortField] = useState<SortField>('timestamp')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [currentPage, setCurrentPage] = useState(1)
  const [showFilters, setShowFilters] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load violations from API
  useEffect(() => {
    const loadViolations = async () => {
      if (violations.length > 0) return // Already loaded

      setLoading(true)
      setError(null)

      try {
        const data = await violationService.getAll()
        setViolations(data)
      } catch (err) {
        console.error('Failed to load violations:', err)
        setError(t('common.loadError'))
      } finally {
        setLoading(false)
      }
    }

    loadViolations()
  }, [violations.length, setViolations, t])

  // Get filtered violations
  const filteredViolations = useMemo(() => {
    return getFilteredViolations()
  }, [getFilteredViolations])

  // Sort violations
  const sortedViolations = useMemo(() => {
    const sorted = [...filteredViolations].sort((a, b) => {
      let comparison = 0

      switch (sortField) {
        case 'timestamp':
          comparison = a.timestamp.getTime() - b.timestamp.getTime()
          break
        case 'type':
          comparison = a.type.localeCompare(b.type)
          break
        case 'status':
          comparison = a.status.localeCompare(b.status)
          break
      }

      return sortDirection === 'asc' ? comparison : -comparison
    })

    return sorted
  }, [filteredViolations, sortField, sortDirection])

  // Pagination
  const totalPages = Math.ceil(sortedViolations.length / ITEMS_PER_PAGE)
  const paginatedViolations = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return sortedViolations.slice(start, start + ITEMS_PER_PAGE)
  }, [sortedViolations, currentPage])

  // Handle filter changes - reset to page 1
  const handleFilterChange = (newFilters: typeof filters) => {
    setFilters(newFilters)
    setCurrentPage(1)
  }

  const handleClearFilters = () => {
    clearFilters()
    setCurrentPage(1)
  }

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const handleStatusChange = async (id: string, status: ViolationStatus) => {
    try {
      await violationService.updateStatus(id, status)
      updateViolationStatus(id, status)
    } catch (err) {
      console.error('Failed to update violation status:', err)
    }
  }

  const handleViewDetail = (id: string) => {
    navigate(`/violations/${id}`)
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 text-neutral-400" />
    }
    return sortDirection === 'asc' ? (
      <ArrowUp className="w-4 h-4 text-primary-500" />
    ) : (
      <ArrowDown className="w-4 h-4 text-primary-500" />
    )
  }

  const activeFiltersCount = Object.values(filters).filter(Boolean).length

  if (loading) {
    return <LoadingState message={t('violations.loading')} />
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
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <AlertTriangle className="w-8 h-8 text-orange-500" />
              {t('pages.violations.title')}
            </h1>
            <p className="text-lg text-gray-600 mt-1">
              {filteredViolations.length} {t('pages.violations.description')}
            </p>
          </div>

          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-3 px-6 py-3 font-semibold rounded-xl transition-all shadow-md hover:shadow-lg transform hover:scale-105 relative ${
              showFilters 
                ? 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white' 
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
            }`}
          >
            <Filter className="w-5 h-5" />
            {t('common.filter')}
            {activeFiltersCount > 0 && (
              <span className="absolute -top-1 -right-1 w-6 h-6 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-bold">
                {activeFiltersCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="animate-fadeIn">
          <ViolationFilters
            filters={filters}
            onFilterChange={handleFilterChange}
            onClearFilters={handleClearFilters}
          />
        </div>
      )}

      {/* Sort Controls */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <div className="flex items-center gap-4">
          <span className="text-lg font-medium text-gray-700">{t('violations.sortBy')}:</span>
          <div className="flex gap-3">
            <button
              onClick={() => handleSort('timestamp')}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
                sortField === 'timestamp' 
                  ? 'bg-blue-100 text-blue-700 shadow-sm' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {t('violations.date')}
              {getSortIcon('timestamp')}
            </button>
            <button
              onClick={() => handleSort('type')}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
                sortField === 'type' 
                  ? 'bg-blue-100 text-blue-700 shadow-sm' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {t('violations.type')}
              {getSortIcon('type')}
            </button>
            <button
              onClick={() => handleSort('status')}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
                sortField === 'status' 
                  ? 'bg-blue-100 text-blue-700 shadow-sm' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {t('violations.status')}
              {getSortIcon('status')}
            </button>
          </div>
        </div>
      </div>

      {/* Violations Grid */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        {paginatedViolations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {paginatedViolations.map((violation, index) => (
              <div
                key={violation.id}
                className="animate-fadeIn"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <ViolationCard
                  violation={violation}
                  onStatusChange={handleStatusChange}
                  onViewDetail={handleViewDetail}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="py-16 text-center">
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-gray-100 flex items-center justify-center">
                <AlertTriangle className="w-8 h-8 text-gray-400" />
              </div>
              <div>
                <p className="text-xl font-semibold text-gray-900">{t('violations.noViolations')}</p>
                <p className="text-gray-600 mt-1">{t('violations.noViolationsHint')}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
          <div className="flex items-center justify-between">
            <p className="text-gray-600">
              {t('violations.showing')} {(currentPage - 1) * ITEMS_PER_PAGE + 1}-
              {Math.min(currentPage * ITEMS_PER_PAGE, sortedViolations.length)} {t('violations.of')}{' '}
              {sortedViolations.length}
            </p>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="p-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-2">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`w-10 h-10 rounded-xl text-sm font-semibold transition-all ${
                      currentPage === page
                        ? 'bg-blue-500 text-white shadow-md'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {page}
                  </button>
                ))}
              </div>

              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="p-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
