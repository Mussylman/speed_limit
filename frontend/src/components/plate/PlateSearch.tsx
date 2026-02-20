/**
 * PlateSearch Component - License Plate Search with Autocomplete
 * Requirements: 3.4 - Plate-based search functionality
 *
 * Aesthetic: Arctic Command Center - Clean search with instant suggestions
 */

import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Search, X, Clock, ChevronRight, Car } from 'lucide-react'
import { useVehicleStore } from '../../stores/vehicleStore'

export interface PlateSearchProps {
  onSelect?: (plate: string) => void
  placeholder?: string
  className?: string
  autoFocus?: boolean
}

// Local storage key for recent searches
const RECENT_SEARCHES_KEY = 'sergek_recent_plate_searches'
const MAX_RECENT_SEARCHES = 5

function getRecentSearches(): string[] {
  try {
    const stored = localStorage.getItem(RECENT_SEARCHES_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

function saveRecentSearch(plate: string): void {
  try {
    const recent = getRecentSearches().filter((p) => p !== plate)
    recent.unshift(plate)
    localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(recent.slice(0, MAX_RECENT_SEARCHES)))
  } catch {
    // Ignore storage errors
  }
}

export function PlateSearch({
  onSelect,
  placeholder,
  className = '',
  autoFocus = false,
}: PlateSearchProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  // Initialize with recent searches directly
  const [recentSearches, setRecentSearches] = useState<string[]>(() => getRecentSearches())
  const [highlightedIndex, setHighlightedIndex] = useState(-1)

  const detections = useVehicleStore((state) => state.detections)

  // Get unique plates from detections
  const uniquePlates = useMemo(() => {
    const plates = new Set<string>()
    detections.forEach((d) => plates.add(d.plate))
    return Array.from(plates)
  }, [detections])

  // Filter suggestions based on query
  const suggestions = useMemo(() => {
    if (!query.trim()) return []
    const normalizedQuery = query.toUpperCase().replace(/\s/g, '')
    return uniquePlates.filter((plate) => plate.toUpperCase().includes(normalizedQuery)).slice(0, 8)
  }, [query, uniquePlates])

  // Combined results: suggestions first, then recent if no query
  const displayItems = useMemo(() => {
    if (query.trim()) {
      return suggestions.map((plate) => ({ plate, isRecent: false }))
    }
    return recentSearches.map((plate) => ({ plate, isRecent: true }))
  }, [query, suggestions, recentSearches])

  // Handle selection
  const handleSelect = useCallback(
    (plate: string) => {
      saveRecentSearch(plate)
      setRecentSearches(getRecentSearches())
      setQuery('')
      setIsOpen(false)
      setHighlightedIndex(-1)

      if (onSelect) {
        onSelect(plate)
      } else {
        navigate(`/vehicles/${plate}`)
      }
    },
    [onSelect, navigate]
  )

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!isOpen) {
        if (e.key === 'ArrowDown' || e.key === 'Enter') {
          setIsOpen(true)
        }
        return
      }

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setHighlightedIndex((prev) => (prev < displayItems.length - 1 ? prev + 1 : 0))
          break
        case 'ArrowUp':
          e.preventDefault()
          setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : displayItems.length - 1))
          break
        case 'Enter':
          e.preventDefault()
          if (highlightedIndex >= 0 && displayItems[highlightedIndex]) {
            handleSelect(displayItems[highlightedIndex].plate)
          } else if (query.trim()) {
            handleSelect(query.trim().toUpperCase())
          }
          break
        case 'Escape':
          setIsOpen(false)
          setHighlightedIndex(-1)
          break
      }
    },
    [isOpen, displayItems, highlightedIndex, query, handleSelect]
  )

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false)
        setHighlightedIndex(-1)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className={`relative ${className}`}>
      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400 pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value.toUpperCase())
            setIsOpen(true)
            setHighlightedIndex(-1)
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || t('search.placeholder')}
          autoFocus={autoFocus}
          className="input pl-10 pr-10 font-mono tracking-wider uppercase"
          aria-label="Search plates"
          aria-expanded={isOpen}
          aria-haspopup="listbox"
          aria-autocomplete="list"
        />
        {query && (
          <button
            onClick={() => {
              setQuery('')
              inputRef.current?.focus()
            }}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-neutral-100 transition-colors"
            aria-label="Clear search"
          >
            <X className="w-4 h-4 text-neutral-400" />
          </button>
        )}
      </div>

      {/* Dropdown */}
      {isOpen && (displayItems.length > 0 || query.trim()) && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-2 bg-white rounded-xl border border-neutral-200 shadow-lg overflow-hidden"
          role="listbox"
        >
          {/* Section header */}
          {displayItems.length > 0 && (
            <div className="px-3 py-2 bg-neutral-50 border-b border-neutral-200">
              <span className="caption text-neutral-500 flex items-center gap-1.5">
                {query.trim() ? (
                  <>
                    <Search className="w-3 h-3" />
                    {t('search.suggestions')}
                  </>
                ) : (
                  <>
                    <Clock className="w-3 h-3" />
                    {t('search.recentSearches')}
                  </>
                )}
              </span>
            </div>
          )}

          {/* Results list */}
          <div className="max-h-64 overflow-y-auto">
            {displayItems.length > 0 ? (
              displayItems.map((item, index) => (
                <SearchResultItem
                  key={item.plate}
                  plate={item.plate}
                  isRecent={item.isRecent}
                  isHighlighted={index === highlightedIndex}
                  query={query}
                  onClick={() => handleSelect(item.plate)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                />
              ))
            ) : query.trim() ? (
              <div className="p-4 text-center">
                <Car className="w-8 h-8 text-neutral-300 mx-auto mb-2" />
                <p className="body-sm text-neutral-500">{t('search.noResults')}</p>
                <button
                  onClick={() => handleSelect(query.trim().toUpperCase())}
                  className="mt-2 text-sm text-primary-600 hover:text-primary-700"
                >
                  Search for "{query.trim().toUpperCase()}"
                </button>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}

// Search result item component
interface SearchResultItemProps {
  plate: string
  isRecent: boolean
  isHighlighted: boolean
  query: string
  onClick: () => void
  onMouseEnter: () => void
}

function SearchResultItem({
  plate,
  isRecent,
  isHighlighted,
  query,
  onClick,
  onMouseEnter,
}: SearchResultItemProps) {
  // Highlight matching text
  const highlightedPlate = useMemo(() => {
    if (!query.trim()) return plate
    const normalizedQuery = query.toUpperCase().replace(/\s/g, '')
    const index = plate.toUpperCase().indexOf(normalizedQuery)
    if (index === -1) return plate

    return (
      <>
        {plate.slice(0, index)}
        <span className="bg-primary-200 text-primary-800 rounded px-0.5">
          {plate.slice(index, index + normalizedQuery.length)}
        </span>
        {plate.slice(index + normalizedQuery.length)}
      </>
    )
  }, [plate, query])

  return (
    <button
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      className={`
        w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors
        ${isHighlighted ? 'bg-primary-50' : 'hover:bg-neutral-50'}
      `}
      role="option"
      aria-selected={isHighlighted}
    >
      {isRecent ? (
        <Clock className="w-4 h-4 text-neutral-400 flex-shrink-0" />
      ) : (
        <Search className="w-4 h-4 text-neutral-400 flex-shrink-0" />
      )}
      <span className="font-mono text-sm tracking-wider flex-1">{highlightedPlate}</span>
      <ChevronRight className="w-4 h-4 text-neutral-400 flex-shrink-0" />
    </button>
  )
}
