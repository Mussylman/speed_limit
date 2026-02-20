/**
 * Vehicle Store - Zustand State Management
 * Requirements: 2.4 (plate detection list)
 */

import { create } from 'zustand'
import type { Vehicle, PlateDetection } from '../types'

interface VehicleState {
  vehicles: Map<string, Vehicle>
  detections: PlateDetection[]
  selectedPlate: string | null
}

interface VehicleActions {
  setVehicles: (vehicles: Vehicle[]) => void
  addVehicle: (vehicle: Vehicle) => void
  addDetection: (detection: PlateDetection) => void
  setDetections: (detections: PlateDetection[]) => void
  setSelectedPlate: (plate: string | null) => void
  getVehicleByPlate: (plate: string) => Vehicle | undefined
  getDetectionsByPlate: (plate: string) => PlateDetection[]
  getDetectionsByCamera: (cameraId: string) => PlateDetection[]
  getRecentDetections: (limit?: number) => PlateDetection[]
}

export type VehicleStore = VehicleState & VehicleActions

export const useVehicleStore = create<VehicleStore>((set, get) => ({
  // State
  vehicles: new Map(),
  detections: [],
  selectedPlate: null,

  // Actions
  setVehicles: (vehicles) =>
    set({
      vehicles: new Map(vehicles.map((v) => [v.plate, v])),
    }),

  addVehicle: (vehicle) =>
    set((state) => {
      const newVehicles = new Map(state.vehicles)
      newVehicles.set(vehicle.plate, vehicle)
      return { vehicles: newVehicles }
    }),

  addDetection: (detection) =>
    set((state) => ({
      detections: [detection, ...state.detections],
    })),

  setDetections: (detections) => set({ detections }),

  setSelectedPlate: (plate) => set({ selectedPlate: plate }),

  getVehicleByPlate: (plate) => get().vehicles.get(plate),

  getDetectionsByPlate: (plate) => get().detections.filter((d) => d.plate === plate),

  getDetectionsByCamera: (cameraId) => get().detections.filter((d) => d.cameraId === cameraId),

  getRecentDetections: (limit = 10) => get().detections.slice(0, limit),
}))
