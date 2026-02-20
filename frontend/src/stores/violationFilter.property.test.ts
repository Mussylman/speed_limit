/**
 * Property Test: Violation Filter Consistency
 * Feature: sergek-camera-system, Property 5: Violation Filter Consistency
 * Validates: Requirements 6.5
 *
 * Property: For any violation filter combination, the filtered results must satisfy
 * ALL active filter criteria simultaneously (conjunction), and removing a filter
 * should return a superset of the previous results.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useViolationStore, applyFilters } from './violationStore'
import type { Violation, ViolationFilters, ViolationType, ViolationStatus } from '../types'

// Arbitrary for generating valid Violation objects
const violationTypeArb: fc.Arbitrary<ViolationType> = fc.constantFrom(
  'speed_limit',
  'red_light',
  'wrong_lane',
  'no_seatbelt',
  'phone_usage',
  'parking',
  'other'
)

const violationStatusArb: fc.Arbitrary<ViolationStatus> = fc.constantFrom(
  'pending',
  'confirmed',
  'dismissed'
)

// Generate dates within a reasonable range (last 30 days)
const dateArb = fc.date({
  min: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
  max: new Date(),
})

// Generate Kazakhstan-style plate numbers
const plateArb = fc
  .tuple(
    fc.integer({ min: 100, max: 999 }),
    fc.array(
      fc.constantFrom(
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z'
      ),
      { minLength: 3, maxLength: 3 }
    ),
    fc.integer({ min: 1, max: 16 })
  )
  .map(([num, letters, region]) => `${num}${letters.join('')}${region.toString().padStart(2, '0')}`)

const violationArb: fc.Arbitrary<Violation> = fc.record({
  id: fc.uuid(),
  type: violationTypeArb,
  plate: plateArb,
  cameraId: fc.string({ minLength: 5, maxLength: 10 }),
  timestamp: dateArb,
  imageUrl: fc.webUrl(),
  videoClipUrl: fc.option(fc.webUrl(), { nil: undefined }),
  status: violationStatusArb,
  fine: fc.option(fc.integer({ min: 5000, max: 100000 }), { nil: undefined }),
  description: fc.option(fc.string({ minLength: 10, maxLength: 100 }), { nil: undefined }),
})

// Arbitrary for generating filter combinations
const filtersArb: fc.Arbitrary<ViolationFilters> = fc.record({
  dateFrom: fc.option(dateArb, { nil: undefined }),
  dateTo: fc.option(dateArb, { nil: undefined }),
  type: fc.option(violationTypeArb, { nil: undefined }),
  plate: fc.option(fc.string({ minLength: 1, maxLength: 5 }), { nil: undefined }),
  status: fc.option(violationStatusArb, { nil: undefined }),
})

// Helper to reset store state
const resetStore = () => {
  useViolationStore.setState({
    violations: [],
    filters: {},
  })
}

// Helper to check if a violation satisfies a filter
const satisfiesFilter = (violation: Violation, filters: ViolationFilters): boolean => {
  if (filters.dateFrom && violation.timestamp < filters.dateFrom) return false
  if (filters.dateTo && violation.timestamp > filters.dateTo) return false
  if (filters.type && violation.type !== filters.type) return false
  if (filters.plate && !violation.plate.includes(filters.plate)) return false
  if (filters.status && violation.status !== filters.status) return false
  return true
}

// Helper to count active filters
const countActiveFilters = (filters: ViolationFilters): number => {
  return Object.values(filters).filter(Boolean).length
}

describe('Property 5: Violation Filter Consistency', () => {
  beforeEach(() => {
    resetStore()
  })

  it('filtered results satisfy ALL active filter criteria (conjunction)', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 0, maxLength: 30 }),
        filtersArb,
        (violations, filters) => {
          // Action: Apply filters
          const filtered = applyFilters(violations, filters)

          // Property: Every filtered violation must satisfy ALL active criteria
          for (const violation of filtered) {
            expect(satisfiesFilter(violation, filters)).toBe(true)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('all violations satisfying filters are included in results', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 0, maxLength: 30 }),
        filtersArb,
        (violations, filters) => {
          // Action: Apply filters
          const filtered = applyFilters(violations, filters)

          // Property: Every violation that satisfies filters should be in results
          for (const violation of violations) {
            if (satisfiesFilter(violation, filters)) {
              const found = filtered.some((v) => v.id === violation.id)
              expect(found).toBe(true)
            }
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('removing a filter returns a superset of previous results', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 30 }),
        filtersArb.filter((f) => countActiveFilters(f) >= 2), // Need at least 2 filters
        (violations, filters) => {
          // Get results with all filters
          const fullFiltered = applyFilters(violations, filters)

          // Remove one filter at a time and verify superset property
          const filterKeys: (keyof ViolationFilters)[] = [
            'dateFrom',
            'dateTo',
            'type',
            'plate',
            'status',
          ]

          for (const key of filterKeys) {
            if (filters[key] !== undefined) {
              // Create filters with one removed
              const reducedFilters = { ...filters, [key]: undefined }
              const reducedFiltered = applyFilters(violations, reducedFilters)

              // Property: Reduced filter results should be a superset
              // Every item in fullFiltered should be in reducedFiltered
              for (const violation of fullFiltered) {
                const found = reducedFiltered.some((v) => v.id === violation.id)
                expect(found).toBe(true)
              }

              // Property: Reduced results should be >= full results
              expect(reducedFiltered.length).toBeGreaterThanOrEqual(fullFiltered.length)
            }
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('empty filters return all violations', () => {
    fc.assert(
      fc.property(fc.array(violationArb, { minLength: 0, maxLength: 30 }), (violations) => {
        // Action: Apply empty filters
        const filtered = applyFilters(violations, {})

        // Property: Should return all violations
        expect(filtered.length).toBe(violations.length)

        // Property: All original violations should be present
        for (const violation of violations) {
          const found = filtered.some((v) => v.id === violation.id)
          expect(found).toBe(true)
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('filtering is idempotent - applying same filter twice gives same result', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 0, maxLength: 30 }),
        filtersArb,
        (violations, filters) => {
          // Action: Apply filters twice
          const result1 = applyFilters(violations, filters)
          const result2 = applyFilters(result1, filters)

          // Property: Second application should not change results
          expect(result2.length).toBe(result1.length)

          for (let i = 0; i < result1.length; i++) {
            expect(result1[i].id).toBe(result2[i].id)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('date range filter works correctly', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 30 }),
        dateArb,
        dateArb,
        (violations, date1, date2) => {
          // Skip if dates are invalid
          if (isNaN(date1.getTime()) || isNaN(date2.getTime())) return true

          // Skip if any violation has invalid timestamp
          const validViolations = violations.filter((v) => !isNaN(v.timestamp.getTime()))
          if (validViolations.length === 0) return true

          // Ensure dateFrom <= dateTo
          const dateFrom = date1 < date2 ? date1 : date2
          const dateTo = date1 < date2 ? date2 : date1

          const filters: ViolationFilters = { dateFrom, dateTo }
          const filtered = applyFilters(validViolations, filters)

          // Property: All filtered violations should be within date range
          for (const violation of filtered) {
            expect(violation.timestamp.getTime()).toBeGreaterThanOrEqual(dateFrom.getTime())
            expect(violation.timestamp.getTime()).toBeLessThanOrEqual(dateTo.getTime())
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('type filter returns only matching violation types', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 30 }),
        violationTypeArb,
        (violations, type) => {
          const filters: ViolationFilters = { type }
          const filtered = applyFilters(violations, filters)

          // Property: All filtered violations should have the specified type
          for (const violation of filtered) {
            expect(violation.type).toBe(type)
          }

          // Property: Count should match manual filter
          const expected = violations.filter((v) => v.type === type)
          expect(filtered.length).toBe(expected.length)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('status filter returns only matching statuses', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 30 }),
        violationStatusArb,
        (violations, status) => {
          const filters: ViolationFilters = { status }
          const filtered = applyFilters(violations, filters)

          // Property: All filtered violations should have the specified status
          for (const violation of filtered) {
            expect(violation.status).toBe(status)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('plate filter performs partial matching', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 30 }),
        fc.string({ minLength: 1, maxLength: 3 }),
        (violations, plateSearch) => {
          const filters: ViolationFilters = { plate: plateSearch }
          const filtered = applyFilters(violations, filters)

          // Property: All filtered violations should contain the search string
          for (const violation of filtered) {
            expect(violation.plate.includes(plateSearch)).toBe(true)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('store getFilteredViolations uses current filters', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 20 }),
        filtersArb,
        (violations, filters) => {
          resetStore()
          const store = useViolationStore.getState()

          // Setup
          store.setViolations(violations)
          store.setFilters(filters)

          // Action
          const storeFiltered = store.getFilteredViolations()
          const directFiltered = applyFilters(violations, filters)

          // Property: Store method should match direct filter application
          expect(storeFiltered.length).toBe(directFiltered.length)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('clearFilters resets to showing all violations', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 5, maxLength: 20 }),
        filtersArb.filter((f) => countActiveFilters(f) >= 1),
        (violations, filters) => {
          resetStore()
          const store = useViolationStore.getState()

          // Setup: Add violations and apply filters
          store.setViolations(violations)
          store.setFilters(filters)

          const filteredCount = store.getFilteredViolations().length

          // Action: Clear filters
          store.clearFilters()

          // Property: Should now return all violations
          const afterClear = store.getFilteredViolations()
          expect(afterClear.length).toBe(violations.length)
          expect(afterClear.length).toBeGreaterThanOrEqual(filteredCount)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
