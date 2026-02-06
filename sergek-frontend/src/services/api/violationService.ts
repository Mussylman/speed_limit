/**
 * Violation Service
 * API methods for violation management
 */

import { apiClient } from './client'
import type { Violation, ViolationFilters, ViolationType, ViolationStatus } from '../../types'

// API response types
interface ViolationResponse {
  id: string
  type: ViolationType
  plate: string
  camera_id: string
  timestamp: string
  image_url: string
  video_clip_url?: string
  status: ViolationStatus
  fine?: number
  description?: string
}

// Transform API response to Violation type
function transformViolation(data: ViolationResponse): Violation {
  return {
    id: data.id,
    type: data.type,
    plate: data.plate,
    cameraId: data.camera_id,
    timestamp: new Date(data.timestamp),
    imageUrl: data.image_url,
    videoClipUrl: data.video_clip_url,
    status: data.status,
    fine: data.fine,
    description: data.description,
  }
}

export const violationService = {
  /**
   * Get all violations with optional filters
   */
  async getAll(filters?: ViolationFilters): Promise<Violation[]> {
    const params = new URLSearchParams()
    
    if (filters?.dateFrom) {
      params.append('date_from', filters.dateFrom.toISOString())
    }
    if (filters?.dateTo) {
      params.append('date_to', filters.dateTo.toISOString())
    }
    if (filters?.type) {
      params.append('type', filters.type)
    }
    if (filters?.plate) {
      params.append('plate', filters.plate)
    }
    if (filters?.status) {
      params.append('status', filters.status)
    }

    const response = await apiClient.get<ViolationResponse[]>(`/violations?${params}`)
    // Backend'den gelen veri array değilse boş array döndür
    const data = Array.isArray(response.data) ? response.data : []
    return data.map(transformViolation)
  },

  /**
   * Get violation by ID
   */
  async getById(id: string): Promise<Violation> {
    const response = await apiClient.get<ViolationResponse>(`/violations/${id}`)
    return transformViolation(response.data)
  },

  /**
   * Update violation status
   */
  async updateStatus(id: string, status: ViolationStatus): Promise<Violation> {
    const response = await apiClient.patch<ViolationResponse>(`/violations/${id}/status`, { status })
    return transformViolation(response.data)
  },

  /**
   * Get violations by plate number
   */
  async getByPlate(plate: string): Promise<Violation[]> {
    const response = await apiClient.get<ViolationResponse[]>(`/violations?plate=${plate}`)
    return response.data.map(transformViolation)
  },

  /**
   * Get violations by camera ID
   */
  async getByCamera(cameraId: string): Promise<Violation[]> {
    const response = await apiClient.get<ViolationResponse[]>(`/violations?camera_id=${cameraId}`)
    return response.data.map(transformViolation)
  },

  /**
   * Get violation statistics
   */
  async getStats(): Promise<{
    total: number
    pending: number
    confirmed: number
    dismissed: number
    byType: Record<ViolationType, number>
  }> {
    const response = await apiClient.get('/violations/stats')
    return response.data
  }
}