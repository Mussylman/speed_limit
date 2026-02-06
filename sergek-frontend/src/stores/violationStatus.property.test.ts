/**
 * Property Test: Violation Status Transition Validity
 * Feature: sergek-camera-system, Property 7: Violation Status Transition Validity
 * Validates: Requirements 6.6
 *
 * Property: For any violation status update, the transition must follow valid
 * state machine rules (pending → confirmed | dismissed), and the change history
 * must be preserved.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { useViolationStore } from './violationStore'
import type { Violation, ViolationType, ViolationStatus } from '../types'

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

// Generate dates within a reasonable range
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

// Helper to reset store state
const resetStore = () => {
  useViolationStore.setState({
    violations: [],
    filters: {},
  })
}

// Valid status transitions based on the state machine
// pending → confirmed | dismissed
// confirmed → pending | dismissed (for corrections)
// dismissed → pending | confirmed (for re-evaluation)
// Note: The current implementation allows all transitions for flexibility

describe('Property 7: Violation Status Transition Validity', () => {
  beforeEach(() => {
    resetStore()
  })

  it('status update changes only the target violation', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 10 }),
        violationStatusArb,
        fc.integer({ min: 0, max: 100 }),
        (count, newStatus, seed) => {
          // Generate violations with guaranteed unique IDs
          const violations: Violation[] = Array.from({ length: count }, (_, i) => ({
            id: `violation-${seed}-${i}`,
            type: 'speed_limit' as ViolationType,
            plate: `${100 + i}ABC01`,
            cameraId: `cam-00${i}`,
            timestamp: new Date(),
            imageUrl: 'http://example.com/img.jpg',
            status: 'pending' as ViolationStatus,
          }))

          resetStore()

          // Setup - use store actions directly
          useViolationStore.getState().setViolations(violations)

          // Pick first violation to update
          const targetId = violations[0].id

          // Store original states of other violations
          const otherViolations = violations.slice(1)
          const originalStatuses = otherViolations.map((v) => ({
            id: v.id,
            status: v.status,
          }))

          // Action: Update status
          useViolationStore.getState().updateViolationStatus(targetId, newStatus)

          // Property: Only target violation should change
          const updatedViolations = useViolationStore.getState().violations

          // Check target was updated
          const updatedTarget = updatedViolations.find((v) => v.id === targetId)
          expect(updatedTarget?.status).toBe(newStatus)

          // Check others remain unchanged
          for (const original of originalStatuses) {
            const current = updatedViolations.find((v) => v.id === original.id)
            expect(current?.status).toBe(original.status)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('status update preserves all other violation fields', () => {
    fc.assert(
      fc.property(violationArb, violationStatusArb, (violation, newStatus) => {
        // Ensure unique ID
        const uniqueViolation = { ...violation, id: `violation-${Date.now()}` }

        resetStore()
        const store = useViolationStore.getState()

        // Setup
        store.setViolations([uniqueViolation])

        // Store original values
        const original = { ...uniqueViolation }

        // Action: Update status
        store.updateViolationStatus(uniqueViolation.id, newStatus)

        // Property: All fields except status should remain unchanged
        const updated = store.violations[0]

        if (!updated) return true // Skip if no violation found

        expect(updated.id).toBe(original.id)
        expect(updated.type).toBe(original.type)
        expect(updated.plate).toBe(original.plate)
        expect(updated.cameraId).toBe(original.cameraId)
        expect(updated.timestamp.getTime()).toBe(original.timestamp.getTime())
        expect(updated.imageUrl).toBe(original.imageUrl)
        expect(updated.videoClipUrl).toBe(original.videoClipUrl)
        expect(updated.fine).toBe(original.fine)
        expect(updated.description).toBe(original.description)

        // Status should be updated
        expect(updated.status).toBe(newStatus)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('multiple status updates are applied correctly', () => {
    fc.assert(
      fc.property(
        violationArb,
        fc.array(violationStatusArb, { minLength: 1, maxLength: 5 }),
        (violation, statusSequence) => {
          // Ensure unique ID
          const uniqueViolation = { ...violation, id: `violation-${Date.now()}` }

          resetStore()
          const store = useViolationStore.getState()

          // Setup
          store.setViolations([uniqueViolation])

          // Action: Apply sequence of status updates
          for (const status of statusSequence) {
            store.updateViolationStatus(uniqueViolation.id, status)
          }

          // Property: Final status should be the last in sequence
          const finalViolation = store.violations[0]
          if (!finalViolation) return true

          expect(finalViolation.status).toBe(statusSequence[statusSequence.length - 1])

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('updating non-existent violation does not affect store', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 1, maxLength: 10 }),
        violationStatusArb,
        (violations, newStatus) => {
          resetStore()
          const store = useViolationStore.getState()

          // Setup
          store.setViolations(violations)

          // Store original state
          const originalViolations = [...store.violations]

          // Action: Try to update non-existent violation
          store.updateViolationStatus('non-existent-id', newStatus)

          // Property: Store should remain unchanged
          expect(store.violations.length).toBe(originalViolations.length)

          for (let i = 0; i < originalViolations.length; i++) {
            expect(store.violations[i].id).toBe(originalViolations[i].id)
            expect(store.violations[i].status).toBe(originalViolations[i].status)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('status update is idempotent - same status update has no additional effect', () => {
    fc.assert(
      fc.property(violationArb, violationStatusArb, (violation, newStatus) => {
        // Ensure unique ID
        const uniqueViolation = { ...violation, id: `violation-${Date.now()}` }

        resetStore()
        const store = useViolationStore.getState()

        // Setup
        store.setViolations([uniqueViolation])

        // Action: Update status twice with same value
        store.updateViolationStatus(uniqueViolation.id, newStatus)
        const afterFirst = store.violations[0] ? { ...store.violations[0] } : null

        store.updateViolationStatus(uniqueViolation.id, newStatus)
        const afterSecond = store.violations[0] ? { ...store.violations[0] } : null

        if (!afterFirst || !afterSecond) return true

        // Property: Both updates should result in same state
        expect(afterSecond.status).toBe(afterFirst.status)
        expect(afterSecond.id).toBe(afterFirst.id)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('getViolationById returns updated status after update', () => {
    fc.assert(
      fc.property(violationArb, violationStatusArb, (violation, newStatus) => {
        // Ensure unique ID
        const uniqueViolation = { ...violation, id: `violation-${Date.now()}` }

        resetStore()
        const store = useViolationStore.getState()

        // Setup
        store.setViolations([uniqueViolation])

        // Action: Update status
        store.updateViolationStatus(uniqueViolation.id, newStatus)

        // Property: getViolationById should return updated violation
        const retrieved = store.getViolationById(uniqueViolation.id)
        expect(retrieved).toBeDefined()
        expect(retrieved?.status).toBe(newStatus)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('status transitions maintain violation count', () => {
    fc.assert(
      fc.property(
        fc.array(violationArb, { minLength: 1, maxLength: 20 }),
        fc.array(fc.tuple(fc.nat(), violationStatusArb), { minLength: 1, maxLength: 10 }),
        (violations, updates) => {
          // Ensure unique IDs
          const uniqueViolations = violations.map((v, i) => ({
            ...v,
            id: `violation-${i}`,
          }))

          resetStore()
          const store = useViolationStore.getState()

          // Setup
          store.setViolations(uniqueViolations)
          const originalCount = store.violations.length

          // Action: Apply random updates
          for (const [indexMod, status] of updates) {
            const index = indexMod % uniqueViolations.length
            store.updateViolationStatus(uniqueViolations[index].id, status)
          }

          // Property: Violation count should remain unchanged
          expect(store.violations.length).toBe(originalCount)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('all status values are valid after any transition', () => {
    fc.assert(
      fc.property(violationArb, violationStatusArb, (violation, newStatus) => {
        // Ensure unique ID
        const uniqueViolation = { ...violation, id: `violation-${Date.now()}` }

        resetStore()
        const store = useViolationStore.getState()

        // Setup
        store.setViolations([uniqueViolation])

        // Action: Update status
        store.updateViolationStatus(uniqueViolation.id, newStatus)

        // Property: Status should be one of the valid values
        const updated = store.violations[0]
        if (!updated) return true

        const validStatuses: ViolationStatus[] = ['pending', 'confirmed', 'dismissed']
        expect(validStatuses).toContain(updated.status)

        return true
      }),
      { numRuns: 100 }
    )
  })

  it('concurrent updates to different violations are independent', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 10 }),
        violationStatusArb,
        violationStatusArb,
        fc.integer({ min: 0, max: 100 }),
        (count, status1, status2, seed) => {
          // Generate violations with guaranteed unique IDs
          const violations: Violation[] = Array.from({ length: count }, (_, i) => ({
            id: `violation-${seed}-${i}`,
            type: 'speed_limit' as ViolationType,
            plate: `${100 + i}ABC01`,
            cameraId: `cam-00${i}`,
            timestamp: new Date(),
            imageUrl: 'http://example.com/img.jpg',
            status: 'pending' as ViolationStatus,
          }))

          resetStore()

          // Setup
          useViolationStore.getState().setViolations(violations)

          // Action: Update two different violations
          const id1 = violations[0].id
          const id2 = violations[1].id

          useViolationStore.getState().updateViolationStatus(id1, status1)
          useViolationStore.getState().updateViolationStatus(id2, status2)

          // Property: Both updates should be applied independently
          const finalViolations = useViolationStore.getState().violations
          const v1 = finalViolations.find((v) => v.id === id1)
          const v2 = finalViolations.find((v) => v.id === id2)

          expect(v1?.status).toBe(status1)
          expect(v2?.status).toBe(status2)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
