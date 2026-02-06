/**
 * Property Test: Map Camera Marker Synchronization
 * Feature: sergek-camera-system, Property 9: Map Camera Marker Synchronization
 * Validates: Requirements 4.2, 4.3, 4.6
 *
 * Property: For any camera location update on the map, the camera's stored
 * coordinates must match the marker position, and clicking the marker must
 * open the correct camera stream.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useCameraStore } from './cameraStore'
import { useMapStore } from './mapStore'
import type { Camera, GeoLocation } from '../types'

// Arbitrary for generating valid GeoLocation (Shymkent area)
const geoLocationArb: fc.Arbitrary<GeoLocation> = fc.record({
  lat: fc.double({ min: 42.2, max: 42.5, noNaN: true }),
  lng: fc.double({ min: 69.4, max: 69.8, noNaN: true }),
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

// Helper to reset store states
const resetStores = () => {
  useCameraStore.setState({
    cameras: [],
    selectedCameraId: null,
    viewMode: 'grid',
  })
  useMapStore.setState({
    center: { lat: 42.3417, lng: 69.5901 },
    zoom: 13,
    isEditMode: false,
    selectedRoute: null,
  })
}

describe('Property 9: Map Camera Marker Synchronization', () => {
  beforeEach(() => {
    resetStores()
  })

  it('camera location update reflects in stored coordinates', () => {
    fc.assert(
      fc.property(cameraArb, geoLocationArb, (camera, newLocation) => {
        resetStores()
        const cameraStore = useCameraStore.getState()

        // Setup: Add camera to store
        cameraStore.addCamera(camera)

        // Action: Update camera location (simulating drag on map)
        cameraStore.updateCamera(camera.id, {
          location: newLocation,
        })

        // Property: Stored coordinates must match the new location
        const updatedCamera = useCameraStore.getState().getCameraById(camera.id)
        expect(updatedCamera).toBeDefined()
        expect(updatedCamera!.location.lat).toBe(newLocation.lat)
        expect(updatedCamera!.location.lng).toBe(newLocation.lng)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('selecting camera by ID returns correct camera for stream', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 15 }), (cameras) => {
        resetStores()
        const cameraStore = useCameraStore.getState()

        // Setup: Add all cameras
        cameraStore.setCameras(cameras)

        // For each camera, verify selection returns correct camera
        for (const camera of cameras) {
          // Action: Select camera (simulating marker click)
          cameraStore.selectCamera(camera.id)

          // Property: Selected camera ID should match
          const state = useCameraStore.getState()
          expect(state.selectedCameraId).toBe(camera.id)

          // Property: getCameraById should return the correct camera
          const selectedCamera = state.getCameraById(camera.id)
          expect(selectedCamera).toBeDefined()
          expect(selectedCamera!.id).toBe(camera.id)
          expect(selectedCamera!.hlsUrl).toBe(camera.hlsUrl)
          expect(selectedCamera!.rtspUrl).toBe(camera.rtspUrl)
        }

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('multiple location updates preserve camera identity', () => {
    fc.assert(
      fc.property(
        cameraArb,
        fc.array(geoLocationArb, { minLength: 1, maxLength: 10 }),
        (camera, locationSequence) => {
          resetStores()
          const cameraStore = useCameraStore.getState()

          // Setup: Add camera
          cameraStore.addCamera(camera)
          const originalId = camera.id
          const originalName = camera.name
          const originalHlsUrl = camera.hlsUrl

          // Action: Apply sequence of location updates
          for (const location of locationSequence) {
            cameraStore.updateCamera(originalId, { location })
          }

          // Property: Camera identity (id, name, urls) should be preserved
          const finalCamera = useCameraStore.getState().getCameraById(originalId)
          expect(finalCamera).toBeDefined()
          expect(finalCamera!.id).toBe(originalId)
          expect(finalCamera!.name).toBe(originalName)
          expect(finalCamera!.hlsUrl).toBe(originalHlsUrl)

          // Final location should match last update
          const lastLocation = locationSequence[locationSequence.length - 1]
          expect(finalCamera!.location.lat).toBe(lastLocation.lat)
          expect(finalCamera!.location.lng).toBe(lastLocation.lng)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('camera count remains constant after location updates', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        fc.nat({ max: 14 }),
        geoLocationArb,
        (cameras, cameraIndex, newLocation) => {
          resetStores()
          const cameraStore = useCameraStore.getState()

          // Setup: Add all cameras
          cameraStore.setCameras(cameras)
          const initialCount = cameras.length

          // Select a valid camera index
          const validIndex = cameraIndex % cameras.length
          const targetCamera = cameras[validIndex]

          // Action: Update one camera's location
          cameraStore.updateCamera(targetCamera.id, { location: newLocation })

          // Property: Camera count should remain the same
          const finalState = useCameraStore.getState()
          expect(finalState.cameras.length).toBe(initialCount)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('edit mode toggle does not affect camera data', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        fc.array(fc.boolean(), { minLength: 1, maxLength: 10 }),
        (cameras, editModeSequence) => {
          resetStores()
          const cameraStore = useCameraStore.getState()
          const mapStore = useMapStore.getState()

          // Setup: Add cameras
          cameraStore.setCameras(cameras)

          // Capture initial camera state
          const initialCameras = JSON.stringify(useCameraStore.getState().cameras)

          // Action: Toggle edit mode multiple times
          for (const shouldBeEdit of editModeSequence) {
            mapStore.setEditMode(shouldBeEdit)
          }

          // Property: Camera data should be unchanged
          const finalCameras = JSON.stringify(useCameraStore.getState().cameras)
          expect(finalCameras).toBe(initialCameras)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('camera removal updates store correctly', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 2, maxLength: 15 }),
        fc.nat({ max: 14 }),
        (cameras, removeIndex) => {
          resetStores()
          const cameraStore = useCameraStore.getState()

          // Setup: Add all cameras
          cameraStore.setCameras(cameras)
          const initialCount = cameras.length

          // Select valid camera to remove
          const validIndex = removeIndex % cameras.length
          const cameraToRemove = cameras[validIndex]

          // Action: Remove camera
          cameraStore.removeCamera(cameraToRemove.id)

          // Property: Camera count should decrease by 1
          const finalState = useCameraStore.getState()
          expect(finalState.cameras.length).toBe(initialCount - 1)

          // Property: Removed camera should not exist
          const removedCamera = finalState.getCameraById(cameraToRemove.id)
          expect(removedCamera).toBeUndefined()

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
