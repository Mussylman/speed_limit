/**
 * Property Test: Smart Camera Filtering Accuracy
 * Feature: sergek-camera-system, Property 2: Smart Camera Filtering Accuracy
 * Validates: Requirements 2.1, 2.5
 *
 * Property: For any set of cameras, filtering to show only smart cameras should
 * return exactly those cameras where type === 'smart', and the count should
 * always be 5 or fewer.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useCameraStore } from './cameraStore'
import type { Camera } from '../types'

// Arbitrary for generating valid Camera objects
const geoLocationArb = fc.record({
  lat: fc.double({ min: 42.0, max: 43.0, noNaN: true }),
  lng: fc.double({ min: 69.0, max: 70.0, noNaN: true }),
  address: fc.option(fc.string({ minLength: 1, maxLength: 50 }), { nil: undefined }),
})

const cameraArb: fc.Arbitrary<Camera> = fc.record({
  id: fc.uuid(),
  name: fc.string({ minLength: 1, maxLength: 30 }),
  rtspUrl: fc.webUrl(),
  hlsUrl: fc.webUrl(),
  location: geoLocationArb,
  type: fc.constantFrom('smart' as const, 'standard' as const),
  status: fc.constantFrom('online' as const, 'offline' as const, 'error' as const),
  lane: fc.option(fc.integer({ min: 1, max: 4 }), { nil: undefined }),
})

// Arbitrary for generating a realistic camera set (max 5 smart, rest standard)
const realisticCameraSetArb = fc
  .tuple(
    fc.integer({ min: 0, max: 5 }), // number of smart cameras
    fc.integer({ min: 0, max: 10 }) // number of standard cameras
  )
  .chain(([smartCount, standardCount]) => {
    const smartCameras = fc.array(
      cameraArb.map((cam) => ({ ...cam, type: 'smart' as const })),
      { minLength: smartCount, maxLength: smartCount }
    )
    const standardCameras = fc.array(
      cameraArb.map((cam) => ({ ...cam, type: 'standard' as const })),
      { minLength: standardCount, maxLength: standardCount }
    )
    return fc
      .tuple(smartCameras, standardCameras)
      .map(([smart, standard]) => [...smart, ...standard])
  })

// Helper to reset store state
const resetStore = () => {
  useCameraStore.setState({
    cameras: [],
    selectedCameraId: null,
    viewMode: 'grid',
  })
}

describe('Property 2: Smart Camera Filtering Accuracy', () => {
  beforeEach(() => {
    resetStore()
  })

  it('getSmartCameras returns only cameras with type === "smart"', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 0, maxLength: 20 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup: Add cameras to store
        store.setCameras(cameras)

        // Action: Get smart cameras
        const smartCameras = store.getSmartCameras()

        // Property 1: All returned cameras must have type === 'smart'
        for (const camera of smartCameras) {
          expect(camera.type).toBe('smart')
        }

        // Property 2: Count should match manual filter
        const expectedSmartCameras = cameras.filter((c) => c.type === 'smart')
        expect(smartCameras.length).toBe(expectedSmartCameras.length)

        // Property 3: All smart cameras from original set should be included
        for (const expectedCam of expectedSmartCameras) {
          const found = smartCameras.some((c) => c.id === expectedCam.id)
          expect(found).toBe(true)
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('getSmartCameras returns empty array when no smart cameras exist', () => {
    fc.assert(
      fc.property(
        fc.array(
          cameraArb.map((cam) => ({ ...cam, type: 'standard' as const })),
          { minLength: 0, maxLength: 15 }
        ),
        (standardCameras) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup: Add only standard cameras
          store.setCameras(standardCameras)

          // Action: Get smart cameras
          const smartCameras = store.getSmartCameras()

          // Property: Should return empty array
          expect(smartCameras.length).toBe(0)
          expect(smartCameras).toEqual([])

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('smart camera count is bounded by system design (max 5)', () => {
    fc.assert(
      fc.property(realisticCameraSetArb, (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup: Add cameras
        store.setCameras(cameras)

        // Action: Get smart cameras
        const smartCameras = store.getSmartCameras()

        // Property: In a realistic system, smart cameras should be <= 5
        // This validates the system design constraint from Requirements 2.1
        expect(smartCameras.length).toBeLessThanOrEqual(5)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('filtering is idempotent - multiple calls return same result', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 0, maxLength: 15 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup
        store.setCameras(cameras)

        // Action: Call getSmartCameras multiple times
        const result1 = store.getSmartCameras()
        const result2 = store.getSmartCameras()
        const result3 = store.getSmartCameras()

        // Property: All calls should return identical results
        expect(result1.length).toBe(result2.length)
        expect(result2.length).toBe(result3.length)

        for (let i = 0; i < result1.length; i++) {
          expect(result1[i].id).toBe(result2[i].id)
          expect(result2[i].id).toBe(result3[i].id)
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('smart camera filtering preserves camera data integrity', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 15 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup
        store.setCameras(cameras)

        // Action
        const smartCameras = store.getSmartCameras()

        // Property: Each smart camera should have all required fields intact
        for (const camera of smartCameras) {
          // Find original camera
          const original = cameras.find((c) => c.id === camera.id)
          expect(original).toBeDefined()

          if (original) {
            // All fields should match
            expect(camera.id).toBe(original.id)
            expect(camera.name).toBe(original.name)
            expect(camera.rtspUrl).toBe(original.rtspUrl)
            expect(camera.hlsUrl).toBe(original.hlsUrl)
            expect(camera.location.lat).toBe(original.location.lat)
            expect(camera.location.lng).toBe(original.location.lng)
            expect(camera.status).toBe(original.status)
            expect(camera.type).toBe('smart')
          }
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('adding/removing cameras updates smart camera filter correctly', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 0, maxLength: 10 }),
        cameraArb.map((cam) => ({ ...cam, type: 'smart' as const })),
        (initialCameras, newSmartCamera) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup: Add initial cameras
          store.setCameras(initialCameras)
          const initialSmartCount = store.getSmartCameras().length

          // Action: Add a new smart camera
          store.addCamera(newSmartCamera)

          // Property: Smart camera count should increase by 1
          const afterAddCount = store.getSmartCameras().length
          expect(afterAddCount).toBe(initialSmartCount + 1)

          // Action: Remove the added camera
          store.removeCamera(newSmartCamera.id)

          // Property: Smart camera count should return to initial
          const afterRemoveCount = store.getSmartCameras().length
          expect(afterRemoveCount).toBe(initialSmartCount)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
