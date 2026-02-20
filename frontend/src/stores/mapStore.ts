/**
 * Map Store - Zustand State Management
 * Requirements: 4.1, 4.2, 4.4 (map and camera positioning)
 */

import { create } from 'zustand'
import type { GeoLocation, VehicleRoute } from '../types'

// Shymkent city center coordinates
const SHYMKENT_CENTER: GeoLocation = {
  lat: 42.3417,
  lng: 69.5901,
  address: 'Shymkent, Kazakhstan',
}

const DEFAULT_ZOOM = 13

interface MapState {
  center: GeoLocation
  zoom: number
  isEditMode: boolean
  selectedRoute: VehicleRoute | null
}

interface MapActions {
  setCenter: (location: GeoLocation) => void
  setZoom: (zoom: number) => void
  toggleEditMode: () => void
  setEditMode: (isEdit: boolean) => void
  setSelectedRoute: (route: VehicleRoute | null) => void
  resetToDefault: () => void
  panTo: (location: GeoLocation) => void
}

export type MapStore = MapState & MapActions

export const useMapStore = create<MapStore>((set) => ({
  // State - Shymkent defaults
  center: SHYMKENT_CENTER,
  zoom: DEFAULT_ZOOM,
  isEditMode: false,
  selectedRoute: null,

  // Actions
  setCenter: (location) => set({ center: location }),

  setZoom: (zoom) => set({ zoom }),

  toggleEditMode: () => set((state) => ({ isEditMode: !state.isEditMode })),

  setEditMode: (isEdit) => set({ isEditMode: isEdit }),

  setSelectedRoute: (route) => set({ selectedRoute: route }),

  resetToDefault: () =>
    set({
      center: SHYMKENT_CENTER,
      zoom: DEFAULT_ZOOM,
      isEditMode: false,
      selectedRoute: null,
    }),

  panTo: (location) => set({ center: location }),
}))

// Export constants for use elsewhere
export { SHYMKENT_CENTER, DEFAULT_ZOOM }
