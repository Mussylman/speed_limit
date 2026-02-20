/**
 * Property Test: Camera CRUD Operations Integrity
 * Feature: sergek-camera-system, Property 6: Camera CRUD Operations Integrity
 * Validates: Requirements 4.4, 4.5, 4.6
 *
 * Property: For any camera add/update/delete operation, the camera list should
 * reflect the change immediately, and the map markers should stay synchronized
 * with the camera store.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useCameraStore } from './cameraStore'
import type { Camera, GeoLocation } from '../types'

// Arbitrary for generating valid GeoLocation objects (Shymkent area)
const geoLocationArb: fc.Arbitrary<GeoLocation> = fc.record({
  lat: fc.double({ min: 42.0, max: 43.0, noNaN: true }),
  lng: fc.double({ min: 69.0, max: 70.0, noNaN: true }),
  address: fc.option(fc.string({ minLength: 1, maxLength: 50 }), { nil: undefined }),
})

// Arbitrary for generating valid Camera objects
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

// Arbitrary for partial camera updates
const cameraUpdateArb: fc.Arbitrary<Partial<Camera>> = fc.record(
  {
    name: fc.string({ minLength: 1, maxLength: 30 }),
    rtspUrl: fc.webUrl(),
    hlsUrl: fc.webUrl(),
    location: geoLocationArb,
    type: fc.constantFrom('smart' as const, 'standard' as const),
    status: fc.constantFrom('online' as const, 'offline' as const, 'error' as const),
  },
  { requiredKeys: [] }
)

// Helper to reset store state
const resetStore = () => {
  useCameraStore.setState({
    cameras: [],
    selectedCameraId: null,
    viewMode: 'grid',
  })
}

describe('Property 6: Camera CRUD Operations Integrity', () => {
  beforeEach(() => {
    resetStore()
  })

  it('adding a camera increases list length by exactly one', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 0, maxLength: 14 }),
        cameraArb,
        (existingCameras, newCamera) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup: Add existing cameras
          store.setCameras(existingCameras)
          const initialCount = useCameraStore.getState().cameras.length

          // Action: Add new camera
          store.addCamera(newCamera)

          // Property: List length should increase by exactly 1
          const finalState = useCameraStore.getState()
          expect(finalState.cameras.length).toBe(initialCount + 1)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('added camera exists in the list with correct data', () => {
    fc.assert(
      fc.property(cameraArb, (newCamera) => {
        resetStore()
        const store = useCameraStore.getState()

        // Action: Add camera
        store.addCamera(newCamera)

        // Property: Camera should exist with correct data
        const finalState = useCameraStore.getState()
        const addedCamera = finalState.cameras.find((c) => c.id === newCamera.id)

        expect(addedCamera).toBeDefined()
        expect(addedCamera?.name).toBe(newCamera.name)
        expect(addedCamera?.rtspUrl).toBe(newCamera.rtspUrl)
        expect(addedCamera?.type).toBe(newCamera.type)
        expect(addedCamera?.location.lat).toBe(newCamera.location.lat)
        expect(addedCamera?.location.lng).toBe(newCamera.location.lng)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('updating a camera preserves list length', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        cameraUpdateArb,
        (cameras, updates) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup
          store.setCameras(cameras)
          const initialCount = useCameraStore.getState().cameras.length
          const targetCamera = cameras[Math.floor(Math.random() * cameras.length)]

          // Action: Update camera
          store.updateCamera(targetCamera.id, updates)

          // Property: List length should remain the same
          const finalState = useCameraStore.getState()
          expect(finalState.cameras.length).toBe(initialCount)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('updating camera location reflects immediately in store', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        geoLocationArb,
        (cameras, newLocation) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup
          store.setCameras(cameras)
          const targetCamera = cameras[Math.floor(Math.random() * cameras.length)]

          // Action: Update camera location
          store.updateCamera(targetCamera.id, { location: newLocation })

          // Property: Location should be updated immediately
          const finalState = useCameraStore.getState()
          const updatedCamera = finalState.cameras.find((c) => c.id === targetCamera.id)

          expect(updatedCamera).toBeDefined()
          expect(updatedCamera?.location.lat).toBe(newLocation.lat)
          expect(updatedCamera?.location.lng).toBe(newLocation.lng)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('deleting a camera decreases list length by exactly one', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 15 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup
        store.setCameras(cameras)
        const initialCount = useCameraStore.getState().cameras.length
        const targetCamera = cameras[Math.floor(Math.random() * cameras.length)]

        // Action: Delete camera
        store.removeCamera(targetCamera.id)

        // Property: List length should decrease by exactly 1
        const finalState = useCameraStore.getState()
        expect(finalState.cameras.length).toBe(initialCount - 1)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('deleted camera no longer exists in the list', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 15 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup
        store.setCameras(cameras)
        const targetCamera = cameras[Math.floor(Math.random() * cameras.length)]

        // Action: Delete camera
        store.removeCamera(targetCamera.id)

        // Property: Camera should no longer exist
        const finalState = useCameraStore.getState()
        const deletedCamera = finalState.cameras.find((c) => c.id === targetCamera.id)

        expect(deletedCamera).toBeUndefined()

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('deleting selected camera clears selection', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 15 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup: Add cameras and select one
        store.setCameras(cameras)
        const targetCamera = cameras[Math.floor(Math.random() * cameras.length)]
        store.selectCamera(targetCamera.id)

        // Verify selection
        expect(useCameraStore.getState().selectedCameraId).toBe(targetCamera.id)

        // Action: Delete the selected camera
        store.removeCamera(targetCamera.id)

        // Property: Selection should be cleared
        const finalState = useCameraStore.getState()
        expect(finalState.selectedCameraId).toBeNull()

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('deleting non-selected camera preserves selection', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 2, maxLength: 15 }), (cameras) => {
        resetStore()
        const store = useCameraStore.getState()

        // Setup: Add cameras and select one
        store.setCameras(cameras)
        const selectedCamera = cameras[0]
        const otherCamera = cameras[1]
        store.selectCamera(selectedCamera.id)

        // Action: Delete a different camera
        store.removeCamera(otherCamera.id)

        // Property: Selection should be preserved
        const finalState = useCameraStore.getState()
        expect(finalState.selectedCameraId).toBe(selectedCamera.id)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('CRUD operations maintain camera ID uniqueness', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 10 }),
        fc.array(cameraArb, { minLength: 1, maxLength: 5 }),
        (initialCameras, newCameras) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup
          store.setCameras(initialCameras)

          // Action: Add multiple cameras
          for (const camera of newCameras) {
            store.addCamera(camera)
          }

          // Property: All camera IDs should be unique
          const finalState = useCameraStore.getState()
          const ids = finalState.cameras.map((c) => c.id)
          const uniqueIds = new Set(ids)

          expect(uniqueIds.size).toBe(ids.length)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('camera locations are valid coordinates after any CRUD operation', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        cameraArb,
        geoLocationArb,
        (cameras, newCamera, newLocation) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup and perform various operations
          store.setCameras(cameras)
          store.addCamera(newCamera)
          if (cameras.length > 0) {
            store.updateCamera(cameras[0].id, { location: newLocation })
          }

          // Property: All cameras should have valid coordinates
          const finalState = useCameraStore.getState()
          for (const camera of finalState.cameras) {
            expect(camera.location.lat).toBeGreaterThanOrEqual(-90)
            expect(camera.location.lat).toBeLessThanOrEqual(90)
            expect(camera.location.lng).toBeGreaterThanOrEqual(-180)
            expect(camera.location.lng).toBeLessThanOrEqual(180)
            expect(Number.isFinite(camera.location.lat)).toBe(true)
            expect(Number.isFinite(camera.location.lng)).toBe(true)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
