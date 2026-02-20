/**
 * Property Test: Camera View Mode Consistency
 * Feature: sergek-camera-system, Property 1: Camera View Mode Consistency
 * Validates: Requirements 1.1, 1.2, 1.4
 *
 * Property: For any camera grid state, when switching between 'grid' and 'single'
 * view modes, the selected camera (if any) should remain selected and visible
 * in the new view mode.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useCameraStore } from './cameraStore'
import type { Camera, CameraViewMode } from '../types'

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

const viewModeArb: fc.Arbitrary<CameraViewMode> = fc.constantFrom('grid', 'single')

// Helper to reset store state
const resetStore = () => {
  useCameraStore.setState({
    cameras: [],
    selectedCameraId: null,
    viewMode: 'grid',
  })
}

describe('Property 1: Camera View Mode Consistency', () => {
  beforeEach(() => {
    resetStore()
  })

  it('selected camera remains selected after view mode switch', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        viewModeArb,
        viewModeArb,
        (cameras, initialMode, targetMode) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup: Add cameras and set initial view mode
          store.setCameras(cameras)
          store.setViewMode(initialMode)

          // Select a random camera from the list
          const selectedCamera = cameras[Math.floor(Math.random() * cameras.length)]
          store.selectCamera(selectedCamera.id)

          // Verify selection before mode switch
          const stateBeforeSwitch = useCameraStore.getState()
          expect(stateBeforeSwitch.selectedCameraId).toBe(selectedCamera.id)

          // Action: Switch view mode
          store.setViewMode(targetMode)

          // Property: Selected camera should remain selected
          const stateAfterSwitch = useCameraStore.getState()
          expect(stateAfterSwitch.selectedCameraId).toBe(selectedCamera.id)
          expect(stateAfterSwitch.viewMode).toBe(targetMode)

          // The selected camera should still exist in the cameras list
          const selectedCameraExists = stateAfterSwitch.cameras.some(
            (c) => c.id === stateAfterSwitch.selectedCameraId
          )
          expect(selectedCameraExists).toBe(true)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('null selection remains null after view mode switch', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 0, maxLength: 15 }),
        viewModeArb,
        viewModeArb,
        (cameras, initialMode, targetMode) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup: Add cameras, set initial mode, no selection
          store.setCameras(cameras)
          store.setViewMode(initialMode)
          store.selectCamera(null)

          // Action: Switch view mode
          store.setViewMode(targetMode)

          // Property: Selection should remain null
          const stateAfterSwitch = useCameraStore.getState()
          expect(stateAfterSwitch.selectedCameraId).toBeNull()
          expect(stateAfterSwitch.viewMode).toBe(targetMode)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('view mode transitions are idempotent for selection state', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        fc.array(viewModeArb, { minLength: 1, maxLength: 10 }),
        (cameras, modeSequence) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup
          store.setCameras(cameras)
          const selectedCamera = cameras[0]
          store.selectCamera(selectedCamera.id)

          // Apply sequence of view mode changes
          for (const mode of modeSequence) {
            store.setViewMode(mode)
          }

          // Property: Selection should be preserved through all transitions
          const finalState = useCameraStore.getState()
          expect(finalState.selectedCameraId).toBe(selectedCamera.id)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('cameras list integrity preserved across view mode changes', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 15 }),
        viewModeArb,
        (cameras, targetMode) => {
          resetStore()
          const store = useCameraStore.getState()

          // Setup
          store.setCameras(cameras)
          const initialCameraCount = cameras.length

          // Action: Switch view mode
          store.setViewMode(targetMode)

          // Property: Camera list should remain unchanged
          const stateAfterSwitch = useCameraStore.getState()
          expect(stateAfterSwitch.cameras.length).toBe(initialCameraCount)

          // All original cameras should still exist
          for (const camera of cameras) {
            const exists = stateAfterSwitch.cameras.some((c) => c.id === camera.id)
            expect(exists).toBe(true)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
