/**
 * Violation Store - Zustand State Management
 * Requirements: 6.5 (violation filtering)
 */

import { create } from 'zustand'
import type { Violation, ViolationFilters } from '../types'

interface ViolationState {
  violations: Violation[]
  filters: ViolationFilters
}

interface ViolationActions {
  setViolations: (violations: Violation[]) => void
  addViolation: (violation: Violation) => void
  updateViolationStatus: (id: string, status: Violation['status']) => void
  setFilters: (filters: ViolationFilters) => void
  clearFilters: () => void
  getFilteredViolations: () => Violation[]
  getViolationById: (id: string) => Violation | undefined
}

export type ViolationStore = ViolationState & ViolationActions

const applyFilters = (violations: Violation[], filters: ViolationFilters): Violation[] => {
  return violations.filter((v) => {
    // Date range filter
    if (filters.dateFrom && v.timestamp < filters.dateFrom) return false
    if (filters.dateTo && v.timestamp > filters.dateTo) return false

    // Type filter
    if (filters.type && v.type !== filters.type) return false

    // Plate filter (partial match)
    if (filters.plate && !v.plate.includes(filters.plate)) return false

    // Status filter
    if (filters.status && v.status !== filters.status) return false

    return true
  })
}

export const useViolationStore = create<ViolationStore>((set, get) => ({
  // State
  violations: [],
  filters: {},

  // Actions
  setViolations: (violations) => set({ violations }),

  addViolation: (violation) =>
    set((state) => ({
      violations: [violation, ...state.violations],
    })),

  updateViolationStatus: (id, status) =>
    set((state) => ({
      violations: state.violations.map((v) => (v.id === id ? { ...v, status } : v)),
    })),

  setFilters: (filters) =>
    set((state) => ({
      filters: { ...state.filters, ...filters },
    })),

  clearFilters: () => set({ filters: {} }),

  getFilteredViolations: () => {
    const { violations, filters } = get()
    return applyFilters(violations, filters)
  },

  getViolationById: (id) => get().violations.find((v) => v.id === id),
}))

// Export filter utility for testing
export { applyFilters }
