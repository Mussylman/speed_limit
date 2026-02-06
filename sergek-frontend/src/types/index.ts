/**
 * Sergek Camera System - Core Type Definitions
 * Requirements: 2.2, 3.2, 5.1, 6.2
 */

// ============================================
// Geographic Types
// ============================================

export interface GeoLocation {
  lat: number
  lng: number
  address?: string
}

// ============================================
// Camera Types
// ============================================

export type CameraType = 'smart' | 'standard'
export type CameraStatus = 'online' | 'offline' | 'error'

export interface Camera {
  id: string
  name: string
  rtspUrl: string
  hlsUrl: string
  location: GeoLocation
  type: CameraType
  status: CameraStatus
  lane?: number
}

// ============================================
// Vehicle & Owner Types
// ============================================

export interface VehicleOwner {
  name: string
  iin: string // Kazakistan Individual Identification Number
  phone?: string
  address?: string
}

export interface Vehicle {
  id: string
  plate: string
  brand: string
  model: string
  color: string
  year: number
  owner: VehicleOwner
}

// ============================================
// Plate Detection Types
// ============================================

export interface PlateDetection {
  id: string
  plate: string
  cameraId: string
  timestamp: Date
  confidence: number
  imageUrl: string
  lane: number
}

// ============================================
// Route Tracking Types
// ============================================

export interface VehicleRoute {
  vehicleId: string
  plate: string
  detections: PlateDetection[]
  routePath: GeoLocation[]
}

// ============================================
// Violation Types
// ============================================

export type ViolationType =
  | 'speed_limit'
  | 'red_light'
  | 'wrong_lane'
  | 'no_seatbelt'
  | 'phone_usage'
  | 'parking'
  | 'other'

export type ViolationStatus = 'pending' | 'confirmed' | 'dismissed'

export interface Violation {
  id: string
  type: ViolationType
  plate: string
  cameraId: string
  timestamp: Date
  imageUrl: string
  videoClipUrl?: string
  status: ViolationStatus
  fine?: number
  description?: string
}

// ============================================
// Filter Types
// ============================================

export interface ViolationFilters {
  dateFrom?: Date
  dateTo?: Date
  type?: ViolationType
  plate?: string
  status?: ViolationStatus
}

// ============================================
// View Mode Types
// ============================================

export type CameraViewMode = 'grid' | 'single'
