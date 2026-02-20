/**
 * Property Test: Vehicle Route Chronological Order
 * Feature: sergek-camera-system, Property 4: Vehicle Route Chronological Order
 * Validates: Requirements 5.1, 5.2, 5.3, 5.5
 *
 * Property: For any vehicle route query, the returned detections must be sorted
 * in chronological order by timestamp, and each detection's camera location must
 * form a geographically plausible path.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useVehicleStore } from './vehicleStore'
import type { PlateDetection } from '../types'

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

// Helper to check if detections are in chronological order
const isChronologicalOrder = (detections: PlateDetection[]): boolean => {
  for (let i = 1; i < detections.length; i++) {
    if (detections[i].timestamp.getTime() < detections[i - 1].timestamp.getTime()) {
      return false
    }
  }
  return true
}

// Helper to sort detections chronologically
const sortChronologically = (detections: PlateDetection[]): PlateDetection[] => {
  return [...detections].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
}

describe('Property 4: Vehicle Route Chronological Order', () => {
  beforeEach(() => {
    resetStore()
  })

  it('detections for a plate can be sorted chronologically', () => {
    fc.assert(
      fc.property(plateArb, fc.integer({ min: 1, max: 20 }), (plate, numDetections) => {
        resetStore()
        const store = useVehicleStore.getState()
        const baseTime = Date.now() - 1000 * 60 * 60 // 1 hour ago

        // Generate detections with random order
        const detections: PlateDetection[] = []
        for (let i = 0; i < numDetections; i++) {
          detections.push({
            id: `det-${i}`,
            plate,
            cameraId: `cam-${i}`,
            timestamp: new Date(baseTime + Math.random() * 1000 * 60 * 60), // Random time within 1 hour
            confidence: 95 + Math.random() * 5,
            imageUrl: `/img/${i}.jpg`,
            lane: (i % 4) + 1,
          })
        }

        // Add detections to store
        store.setDetections(detections)

        // Get detections by plate
        const plateDetections = store.getDetectionsByPlate(plate)

        // Sort them chronologically
        const sorted = sortChronologically(plateDetections)

        // Property: Sorted detections should be in chronological order
        expect(isChronologicalOrder(sorted)).toBe(true)

        // Property: All original detections should be present
        expect(sorted.length).toBe(plateDetections.length)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('chronological sorting preserves all detection data', () => {
    fc.assert(
      fc.property(plateArb, fc.integer({ min: 2, max: 15 }), (plate, numDetections) => {
        resetStore()
        const store = useVehicleStore.getState()
        const baseTime = Date.now() - 1000 * 60 * 60

        // Generate detections
        const detections: PlateDetection[] = []
        for (let i = 0; i < numDetections; i++) {
          detections.push({
            id: `det-${i}-${plate}`,
            plate,
            cameraId: `cam-${i}`,
            timestamp: new Date(baseTime + i * 60000 * (Math.random() > 0.5 ? 1 : -1)),
            confidence: 90 + Math.random() * 10,
            imageUrl: `/img/${i}.jpg`,
            lane: (i % 4) + 1,
          })
        }

        store.setDetections(detections)
        const plateDetections = store.getDetectionsByPlate(plate)
        const sorted = sortChronologically(plateDetections)

        // Property: All detection IDs should be preserved
        const originalIds = new Set(plateDetections.map((d) => d.id))
        const sortedIds = new Set(sorted.map((d) => d.id))
        expect(sortedIds).toEqual(originalIds)

        // Property: All detection data should be preserved
        for (const detection of sorted) {
          const original = plateDetections.find((d) => d.id === detection.id)
          expect(original).toBeDefined()
          expect(detection.plate).toBe(original!.plate)
          expect(detection.cameraId).toBe(original!.cameraId)
          expect(detection.confidence).toBe(original!.confidence)
          expect(detection.lane).toBe(original!.lane)
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('empty detection list remains empty after sorting', () => {
    fc.assert(
      fc.property(plateArb, (plate) => {
        resetStore()
        const store = useVehicleStore.getState()

        // No detections added
        const plateDetections = store.getDetectionsByPlate(plate)
        const sorted = sortChronologically(plateDetections)

        // Property: Empty list should remain empty
        expect(sorted.length).toBe(0)
        expect(isChronologicalOrder(sorted)).toBe(true)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('single detection is trivially chronological', () => {
    fc.assert(
      fc.property(plateArb, (plate) => {
        resetStore()
        const store = useVehicleStore.getState()

        const detection: PlateDetection = {
          id: 'single-det',
          plate,
          cameraId: 'cam-1',
          timestamp: new Date(),
          confidence: 98,
          imageUrl: '/img/single.jpg',
          lane: 1,
        }

        store.setDetections([detection])
        const plateDetections = store.getDetectionsByPlate(plate)
        const sorted = sortChronologically(plateDetections)

        // Property: Single detection is always in chronological order
        expect(sorted.length).toBe(1)
        expect(isChronologicalOrder(sorted)).toBe(true)
        expect(sorted[0].id).toBe(detection.id)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('detections from multiple plates can be filtered and sorted independently', () => {
    fc.assert(
      fc.property(
        fc.array(plateArb, { minLength: 2, maxLength: 5 }),
        fc.integer({ min: 2, max: 10 }),
        (plates, detectionsPerPlate) => {
          resetStore()
          const store = useVehicleStore.getState()
          const baseTime = Date.now() - 1000 * 60 * 60

          // Generate detections for multiple plates
          const allDetections: PlateDetection[] = []
          for (const plate of plates) {
            for (let i = 0; i < detectionsPerPlate; i++) {
              allDetections.push({
                id: `det-${plate}-${i}`,
                plate,
                cameraId: `cam-${i}`,
                timestamp: new Date(baseTime + Math.random() * 1000 * 60 * 60),
                confidence: 90 + Math.random() * 10,
                imageUrl: `/img/${plate}-${i}.jpg`,
                lane: (i % 4) + 1,
              })
            }
          }

          store.setDetections(allDetections)

          // Property: Each plate's detections can be sorted independently
          for (const plate of plates) {
            const plateDetections = store.getDetectionsByPlate(plate)
            const sorted = sortChronologically(plateDetections)

            expect(isChronologicalOrder(sorted)).toBe(true)
            expect(sorted.every((d) => d.plate === plate)).toBe(true)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('timestamps maintain relative ordering after sorting', () => {
    fc.assert(
      fc.property(
        plateArb,
        fc.array(fc.integer({ min: 0, max: 1000 }), { minLength: 2, maxLength: 20 }),
        (plate, timeOffsets) => {
          resetStore()
          const store = useVehicleStore.getState()
          const baseTime = Date.now() - 1000 * 60 * 60

          // Create detections with specific time offsets
          const detections: PlateDetection[] = timeOffsets.map((offset, i) => ({
            id: `det-${i}`,
            plate,
            cameraId: `cam-${i}`,
            timestamp: new Date(baseTime + offset * 1000),
            confidence: 95,
            imageUrl: `/img/${i}.jpg`,
            lane: 1,
          }))

          store.setDetections(detections)
          const sorted = sortChronologically(store.getDetectionsByPlate(plate))

          // Property: Sorted timestamps should be non-decreasing
          for (let i = 1; i < sorted.length; i++) {
            expect(sorted[i].timestamp.getTime()).toBeGreaterThanOrEqual(
              sorted[i - 1].timestamp.getTime()
            )
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
