/**
 * Vehicle Service
 * API methods for vehicle and plate detection management
 */

import { apiClient } from './client'
import type { Vehicle, PlateDetection } from '../../types'

// API response types
interface VehicleResponse {
  id: string
  plate: string
  brand: string
  model: string
  color: string
  year: number
  owner: {
    name: string
    iin: string
    phone?: string
    address?: string
  }
}

interface PlateDetectionResponse {
  id: string
  plate: string
  camera_id: string
  timestamp: string
  confidence: number
  image_url: string
  lane: number
}

// Transform API responses
function transformVehicle(data: VehicleResponse): Vehicle {
  return {
    id: data.id,
    plate: data.plate,
    brand: data.brand,
    model: data.model,
    color: data.color,
    year: data.year,
    owner: data.owner,
  }
}

function transformDetection(data: PlateDetectionResponse): PlateDetection {
  return {
    id: data.id,
    plate: data.plate,
    cameraId: data.camera_id,
    timestamp: new Date(data.timestamp),
    confidence: data.confidence,
    imageUrl: data.image_url,
    lane: data.lane,
  }
}

export const vehicleService = {
  /**
   * Get vehicle by plate number
   */
  async getByPlate(plate: string): Promise<Vehicle> {
    const response = await apiClient.get<VehicleResponse>(`/vehicles/${encodeURIComponent(plate)}`)
    return transformVehicle(response.data)
  },

  /**
   * Search vehicles by partial plate
   */
  async search(query: string): Promise<Vehicle[]> {
    const response = await apiClient.get<VehicleResponse[]>(`/vehicles/search?q=${encodeURIComponent(query)}`)
    return response.data.map(transformVehicle)
  },

  /**
   * Get all plate detections
   */
  async getDetections(limit?: number): Promise<PlateDetection[]> {
    const params = limit ? `?limit=${limit}` : ''
    const response = await apiClient.get<PlateDetectionResponse[]>(`/detections${params}`)
    // Backend'den gelen veri array değilse boş array döndür
    const data = Array.isArray(response.data) ? response.data : []
    return data.map(transformDetection)
  },

  /**
   * Get detections by plate number
   */
  async getDetectionsByPlate(plate: string): Promise<PlateDetection[]> {
    const response = await apiClient.get<PlateDetectionResponse[]>(`/detections?plate=${encodeURIComponent(plate)}`)
    return response.data.map(transformDetection)
  },

  /**
   * Get detections by camera ID
   */
  async getDetectionsByCamera(cameraId: string): Promise<PlateDetection[]> {
    const response = await apiClient.get<PlateDetectionResponse[]>(`/detections?camera_id=${cameraId}`)
    return response.data.map(transformDetection)
  },

  /**
   * Get recent detections for smart cameras
   */
  async getRecentDetections(limit: number = 50): Promise<PlateDetection[]> {
    const response = await apiClient.get<PlateDetectionResponse[]>(`/detections/recent?limit=${limit}`)
    return response.data.map(transformDetection)
  },

  /**
   * Get detection statistics
   */
  async getDetectionStats(): Promise<{
    total: number
    today: number
    thisWeek: number
    thisMonth: number
    byCamera: Record<string, number>
  }> {
    const response = await apiClient.get('/detections/stats')
    return response.data
  }
}
