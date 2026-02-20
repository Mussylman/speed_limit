/**
 * Camera Store - Zustand State Management
 * Requirements: 1.4 (view modes)
 */

import { create } from 'zustand'
import type { Camera, CameraViewMode } from '../types'

interface CameraState {
  cameras: Camera[]
  selectedCameraId: string | null
  viewMode: CameraViewMode
}

interface CameraActions {
  setCameras: (cameras: Camera[]) => void
  selectCamera: (id: string | null) => void
  setViewMode: (mode: CameraViewMode) => void
  addCamera: (camera: Camera) => void
  updateCamera: (id: string, updates: Partial<Camera>) => void
  removeCamera: (id: string) => void
  getSmartCameras: () => Camera[]
  getCameraById: (id: string) => Camera | undefined
}

export type CameraStore = CameraState & CameraActions

export const useCameraStore = create<CameraStore>((set, get) => ({
  // State
  cameras: [],
  selectedCameraId: null,
  viewMode: 'grid',

  // Actions
  setCameras: (cameras) => set({ cameras }),

  selectCamera: (id) => set({ selectedCameraId: id }),

  setViewMode: (mode) => set({ viewMode: mode }),

  addCamera: (camera) =>
    set((state) => ({
      cameras: [...state.cameras, camera],
    })),

  updateCamera: (id, updates) =>
    set((state) => ({
      cameras: state.cameras.map((cam) => (cam.id === id ? { ...cam, ...updates } : cam)),
    })),

  removeCamera: (id) =>
    set((state) => ({
      cameras: state.cameras.filter((cam) => cam.id !== id),
      selectedCameraId: state.selectedCameraId === id ? null : state.selectedCameraId,
    })),

  getSmartCameras: () => get().cameras.filter((cam) => cam.type === 'smart'),

  getCameraById: (id) => get().cameras.find((cam) => cam.id === id),
}))
