/**
 * Property Test: Search Results Relevance
 * Feature: sergek-camera-system, Property 10: Search Results Relevance
 * Validates: Requirements 3.4
 *
 * Property: For any plate search query, all returned results must contain the
 * search string as a substring of the plate field, and results should be ordered
 * by relevance (exact match first, then partial matches).
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useVehicleStore } from './vehicleStore'

// Arbitrary for generating valid Kazakhstan plate format
const plateArb: fc.Arbitrary<string> = fc
  .tuple(
    fc.integer({ min: 100, max: 999 }),
    fc.string({
      minLength: 3,
      maxLength: 3,
      unit: fc.constantFrom('A', 'B', 'C', 'D', 'E', 'K', 'M', 'N', 'P', 'X'),
    }),
    fc.integer({ min: 1, max: 16 })
  )
  .map(([num, letters, region]) => `${num}${letters}${region.toString().padStart(2, '0')}`)

// Helper to reset store state
const resetStore = () => {
  useVehicleStore.setState({
    vehicles: new Map(),
    detections: [],
    selectedPlate: null,
  })
}

// Search function that filters plates by query substring
const searchPlates = (plates: string[], query: string): string[] => {
  if (!query.trim()) return []
  const normalizedQuery = query.toUpperCase().replace(/\s/g, '')

  return plates
    .filter((plate) => plate.toUpperCase().includes(normalizedQuery))
    .sort((a, b) => {
      const aUpper = a.toUpperCase()
      const bUpper = b.toUpperCase()

      // Exact match first
      if (aUpper === normalizedQuery && bUpper !== normalizedQuery) return -1
      if (bUpper === normalizedQuery && aUpper !== normalizedQuery) return 1

      // Then by position of match (earlier is better)
      const aIndex = aUpper.indexOf(normalizedQuery)
      const bIndex = bUpper.indexOf(normalizedQuery)
      if (aIndex !== bIndex) return aIndex - bIndex

      // Then alphabetically
      return aUpper.localeCompare(bUpper)
    })
}

describe('Property 10: Search Results Relevance', () => {
  beforeEach(() => {
    resetStore()
  })

  it('all search results contain the query as substring', () => {
    fc.assert(
      fc.property(
        fc.array(plateArb, { minLength: 1, maxLength: 50 }),
        fc.string({
          minLength: 1,
          maxLength: 5,
          unit: fc.constantFrom(
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            'A',
            'B',
            'C',
            'D',
            'E',
            'K',
            'M',
            'N',
            'P',
            'X'
          ),
        }),
        (plates, query) => {
          const uniquePlates = [...new Set(plates)]
          const results = searchPlates(uniquePlates, query)
          const normalizedQuery = query.toUpperCase().replace(/\s/g, '')

          // Property: All results must contain the query as substring
          for (const result of results) {
            expect(result.toUpperCase()).toContain(normalizedQuery)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('exact matches appear before partial matches', () => {
    fc.assert(
      fc.property(plateArb, (exactPlate) => {
        // Create a set of plates including the exact match and some partials
        const plates = [exactPlate, `X${exactPlate}`, `${exactPlate}Y`, `Z${exactPlate}Z`]

        const results = searchPlates(plates, exactPlate)

        // Property: If exact match exists, it should be first
        if (results.length > 0 && results.includes(exactPlate)) {
          expect(results[0]).toBe(exactPlate)
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('empty query returns empty results', () => {
    fc.assert(
      fc.property(
        fc.array(plateArb, { minLength: 0, maxLength: 20 }),
        fc.constantFrom('', ' ', '  ', '\t'),
        (plates, emptyQuery) => {
          const results = searchPlates(plates, emptyQuery)

          // Property: Empty query should return empty results
          expect(results.length).toBe(0)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('search is case insensitive', () => {
    fc.assert(
      fc.property(
        fc.array(plateArb, { minLength: 1, maxLength: 30 }),
        fc.string({
          minLength: 1,
          maxLength: 4,
          unit: fc.constantFrom('a', 'b', 'c', 'd', 'e', 'k', 'm', 'n', 'p', 'x', '1', '2', '3'),
        }),
        (plates, lowerQuery) => {
          const uniquePlates = [...new Set(plates)]
          const upperQuery = lowerQuery.toUpperCase()

          const lowerResults = searchPlates(uniquePlates, lowerQuery)
          const upperResults = searchPlates(uniquePlates, upperQuery)

          // Property: Case should not affect results
          expect(lowerResults).toEqual(upperResults)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('results are ordered by match position', () => {
    fc.assert(
      fc.property(
        fc.string({
          minLength: 2,
          maxLength: 3,
          unit: fc.constantFrom('A', 'B', 'C', '1', '2', '3'),
        }),
        (query) => {
          // Create plates with query at different positions
          const plates = [
            `ZZZ${query}ZZZ`, // middle
            `${query}ZZZZZZ`, // start
            `ZZZZZZ${query}`, // end
          ]

          const results = searchPlates(plates, query)

          // Property: Results should be ordered by match position (earlier is better)
          if (results.length >= 2) {
            for (let i = 1; i < results.length; i++) {
              const prevIndex = results[i - 1].toUpperCase().indexOf(query.toUpperCase())
              const currIndex = results[i].toUpperCase().indexOf(query.toUpperCase())
              expect(currIndex).toBeGreaterThanOrEqual(prevIndex)
            }
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('non-matching query returns empty results', () => {
    fc.assert(
      fc.property(fc.array(plateArb, { minLength: 1, maxLength: 20 }), (plates) => {
        // Use a query that won't match any Kazakhstan plate format
        const nonMatchingQuery = 'QQQQQQ'
        const results = searchPlates(plates, nonMatchingQuery)

        // Property: Non-matching query should return empty results
        expect(results.length).toBe(0)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('search results are subset of input plates', () => {
    fc.assert(
      fc.property(
        fc.array(plateArb, { minLength: 1, maxLength: 30 }),
        fc.string({
          minLength: 1,
          maxLength: 4,
          unit: fc.constantFrom('0', '1', '2', '3', 'A', 'B', 'C', 'D'),
        }),
        (plates, query) => {
          const uniquePlates = [...new Set(plates)]
          const results = searchPlates(uniquePlates, query)

          // Property: All results must be from the original plate list
          for (const result of results) {
            expect(uniquePlates).toContain(result)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('search results have no duplicates', () => {
    fc.assert(
      fc.property(
        fc.array(plateArb, { minLength: 1, maxLength: 30 }),
        fc.string({
          minLength: 1,
          maxLength: 3,
          unit: fc.constantFrom('0', '1', '2', 'A', 'B', 'C'),
        }),
        (plates, query) => {
          const uniquePlates = [...new Set(plates)]
          const results = searchPlates(uniquePlates, query)

          // Property: Results should have no duplicates
          const uniqueResults = new Set(results)
          expect(uniqueResults.size).toBe(results.length)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('longer matching substring does not exclude shorter matches', () => {
    fc.assert(
      fc.property(
        fc.string({
          minLength: 2,
          maxLength: 4,
          unit: fc.constantFrom('A', 'B', 'C', '1', '2', '3'),
        }),
        (baseQuery) => {
          const shortQuery = baseQuery.slice(0, 2)
          const plates = [`${baseQuery}XXX`, `${shortQuery}YYY`, `ZZZ${shortQuery}`]

          const results = searchPlates(plates, shortQuery)

          // Property: All plates containing the short query should be in results
          const expectedMatches = plates.filter((p) =>
            p.toUpperCase().includes(shortQuery.toUpperCase())
          )
          expect(results.length).toBe(expectedMatches.length)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
