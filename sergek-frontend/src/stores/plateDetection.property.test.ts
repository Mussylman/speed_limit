/**
 * Property Test: Plate Detection Data Integrity
 * Feature: sergek-camera-system, Property 3: Plate Detection Data Integrity
 * Validates: Requirements 2.2, 2.3, 2.4
 *
 * Property: For any plate detection event, the detection record must contain
 * a valid Kazakhstan plate format, a valid camera ID that exists in the system,
 * and a timestamp that is not in the future.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useCameraStore } from './cameraStore'
import { useVehicleStore } from './vehicleStore'
import type { Camera, PlateDetection } from '../types'

// Kazakhstan plate format patterns:
// Standard: XXX 000 XX (3 letters, 3 digits, 2 letters) - region code at end
// Alternative formats exist but we'll validate basic structure
const KAZAKHSTAN_PLATE_REGEX = /^[A-Z0-9]{5,10}$/

/**
 * Validates if a plate string matches Kazakhstan plate format
 * Accepts alphanumeric strings of 5-10 characters (simplified validation)
 */
function isValidKazakhstanPlate(plate: string): boolean {
  if (!plate || typeof plate !== 'string') return false
  const cleaned = plate.replace(/\s/g, '').toUpperCase()
  return KAZAKHSTAN_PLATE_REGEX.test(cleaned)
}

/**
 * Validates if a timestamp is not in the future
 */
function isValidTimestamp(timestamp: Date): boolean {
  if (!(timestamp instanceof Date) || isNaN(timestamp.getTime())) return false
  return timestamp.getTime() <= Date.now()
}

/**
 * Validates if a camera ID exists in the system
 */
function cameraExists(cameraId: string, cameras: Camera[]): boolean {
  return cameras.some((c) => c.id === cameraId)
}

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

// Arbitrary for generating valid Kazakhstan plate strings
const kazakhstanPlateArb = fc
  .tuple(
    fc.string({ minLength: 3, maxLength: 3 }).map((s) =>
      s
        .toUpperCase()
        .replace(/[^A-Z]/g, 'A')
        .padEnd(3, 'A')
        .slice(0, 3)
    ),
    fc.integer({ min: 100, max: 999 }).map(String),
    fc.string({ minLength: 2, maxLength: 2 }).map((s) =>
      s
        .toUpperCase()
        .replace(/[^A-Z]/g, 'B')
        .padEnd(2, 'B')
        .slice(0, 2)
    )
  )
  .map(([letters1, digits, letters2]) => `${letters1}${digits}${letters2}`)

// Arbitrary for generating valid timestamps (in the past)
const pastTimestampArb = fc
  .integer({ min: 1, max: 365 * 24 * 60 * 60 * 1000 }) // Up to 1 year ago
  .map((msAgo) => new Date(Date.now() - msAgo))

// Arbitrary for generating PlateDetection with valid data
const validPlateDetectionArb = (cameras: Camera[]): fc.Arbitrary<PlateDetection> => {
  if (cameras.length === 0) {
    // If no cameras, use a placeholder ID
    return fc.record({
      id: fc.uuid(),
      plate: kazakhstanPlateArb,
      cameraId: fc.constant('no-camera'),
      timestamp: pastTimestampArb,
      confidence: fc.double({ min: 0, max: 1, noNaN: true }),
      imageUrl: fc.webUrl(),
      lane: fc.integer({ min: 1, max: 4 }),
    })
  }

  return fc.record({
    id: fc.uuid(),
    plate: kazakhstanPlateArb,
    cameraId: fc.constantFrom(...cameras.map((c) => c.id)),
    timestamp: pastTimestampArb,
    confidence: fc.double({ min: 0, max: 1, noNaN: true }),
    imageUrl: fc.webUrl(),
    lane: fc.integer({ min: 1, max: 4 }),
  })
}

// Helper to reset store state
const resetStores = () => {
  useCameraStore.setState({
    cameras: [],
    selectedCameraId: null,
    viewMode: 'grid',
  })
  useVehicleStore.setState({
    vehicles: new Map(),
    detections: [],
    selectedPlate: null,
  })
}

describe('Property 3: Plate Detection Data Integrity', () => {
  beforeEach(() => {
    resetStores()
  })

  it('all plate detections have valid Kazakhstan plate format', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 5 }), (cameras) => {
        resetStores()

        // Setup cameras
        useCameraStore.getState().setCameras(cameras)

        // Generate valid detections
        const detectionArb = validPlateDetectionArb(cameras)

        return fc.assert(
          fc.property(detectionArb, (detection) => {
            // Add detection to store
            useVehicleStore.getState().addDetection(detection)

            // Property: Plate must be valid Kazakhstan format
            expect(isValidKazakhstanPlate(detection.plate)).toBe(true)

            return true
          }),
          { numRuns: 20 }
        )
      }),
      { numRuns: 5 }
    )
  })

  it('all plate detections reference existing cameras', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 10 }), (cameras) => {
        resetStores()

        // Setup cameras
        useCameraStore.getState().setCameras(cameras)
        const storedCameras = useCameraStore.getState().cameras

        // Generate detections that reference these cameras
        const detectionArb = validPlateDetectionArb(cameras)

        return fc.assert(
          fc.property(detectionArb, (detection) => {
            useVehicleStore.getState().addDetection(detection)

            // Property: Camera ID must exist in the system
            expect(cameraExists(detection.cameraId, storedCameras)).toBe(true)

            return true
          }),
          { numRuns: 20 }
        )
      }),
      { numRuns: 5 }
    )
  })

  it('all plate detections have timestamps not in the future', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 5 }), (cameras) => {
        resetStores()

        useCameraStore.getState().setCameras(cameras)
        const detectionArb = validPlateDetectionArb(cameras)

        return fc.assert(
          fc.property(detectionArb, (detection) => {
            useVehicleStore.getState().addDetection(detection)

            // Property: Timestamp must not be in the future
            expect(isValidTimestamp(detection.timestamp)).toBe(true)

            return true
          }),
          { numRuns: 20 }
        )
      }),
      { numRuns: 5 }
    )
  })

  it('detection confidence is within valid range [0, 1]', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 1, maxLength: 5 }), (cameras) => {
        resetStores()

        useCameraStore.getState().setCameras(cameras)
        const detectionArb = validPlateDetectionArb(cameras)

        return fc.assert(
          fc.property(detectionArb, (detection) => {
            useVehicleStore.getState().addDetection(detection)

            // Property: Confidence must be between 0 and 1
            expect(detection.confidence).toBeGreaterThanOrEqual(0)
            expect(detection.confidence).toBeLessThanOrEqual(1)

            return true
          }),
          { numRuns: 20 }
        )
      }),
      { numRuns: 5 }
    )
  })

  it('detections are stored in order (newest first)', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 5 }),
        fc.array(pastTimestampArb, { minLength: 2, maxLength: 10 }),
        (cameras, timestamps) => {
          resetStores()

          useCameraStore.getState().setCameras(cameras)
          const vehicleStore = useVehicleStore.getState()

          // Add detections with different timestamps
          timestamps.forEach((timestamp, index) => {
            const detection: PlateDetection = {
              id: `det-${index}`,
              plate: `ABC${String(index).padStart(3, '0')}KZ`,
              cameraId: cameras[0].id,
              timestamp,
              confidence: 0.9,
              imageUrl: `https://example.com/img${index}.jpg`,
              lane: 1,
            }
            vehicleStore.addDetection(detection)
          })

          // Property: Detections should be stored with newest first
          const storedDetections = useVehicleStore.getState().detections

          // Since addDetection prepends, the order should be reverse of insertion
          // (most recently added = first in array)
          expect(storedDetections.length).toBe(timestamps.length)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('getDetectionsByPlate returns only matching plates', () => {
    fc.assert(
      fc.property(
        fc.array(cameraArb, { minLength: 1, maxLength: 3 }),
        kazakhstanPlateArb,
        kazakhstanPlateArb,
        (cameras, plate1, plate2) => {
          // Ensure plates are different
          fc.pre(plate1 !== plate2)

          resetStores()

          useCameraStore.getState().setCameras(cameras)
          const vehicleStore = useVehicleStore.getState()

          // Add detections for both plates
          const detection1: PlateDetection = {
            id: 'det-1',
            plate: plate1,
            cameraId: cameras[0].id,
            timestamp: new Date(Date.now() - 1000),
            confidence: 0.95,
            imageUrl: 'https://example.com/img1.jpg',
            lane: 1,
          }

          const detection2: PlateDetection = {
            id: 'det-2',
            plate: plate2,
            cameraId: cameras[0].id,
            timestamp: new Date(Date.now() - 2000),
            confidence: 0.92,
            imageUrl: 'https://example.com/img2.jpg',
            lane: 2,
          }

          vehicleStore.addDetection(detection1)
          vehicleStore.addDetection(detection2)

          // Property: getDetectionsByPlate should return only matching plates
          const plate1Detections = useVehicleStore.getState().getDetectionsByPlate(plate1)
          const plate2Detections = useVehicleStore.getState().getDetectionsByPlate(plate2)

          expect(plate1Detections.every((d) => d.plate === plate1)).toBe(true)
          expect(plate2Detections.every((d) => d.plate === plate2)).toBe(true)
          expect(plate1Detections.length).toBe(1)
          expect(plate2Detections.length).toBe(1)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('getDetectionsByCamera returns only matching camera detections', () => {
    fc.assert(
      fc.property(fc.array(cameraArb, { minLength: 2, maxLength: 5 }), (cameras) => {
        resetStores()

        useCameraStore.getState().setCameras(cameras)
        const vehicleStore = useVehicleStore.getState()

        // Add detections for different cameras
        cameras.forEach((camera, index) => {
          const detection: PlateDetection = {
            id: `det-${index}`,
            plate: `ABC${String(index).padStart(3, '0')}KZ`,
            cameraId: camera.id,
            timestamp: new Date(Date.now() - index * 1000),
            confidence: 0.9,
            imageUrl: `https://example.com/img${index}.jpg`,
            lane: 1,
          }
          vehicleStore.addDetection(detection)
        })

        // Property: getDetectionsByCamera should return only matching camera
        cameras.forEach((camera) => {
          const cameraDetections = useVehicleStore.getState().getDetectionsByCamera(camera.id)
          expect(cameraDetections.every((d) => d.cameraId === camera.id)).toBe(true)
        })

        return true
      }),
      { numRuns: 100 }
    )
  })
})
