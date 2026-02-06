/**
 * Camera Service
 * API methods for camera CRUD operations and streaming
 */

import { apiClient } from './client'
import type { Camera, GeoLocation, CameraType } from '../../types'

// Camera form data for create/update operations
export interface CameraFormData {
  name: string
  rtspUrl: string
  type: CameraType
  location: GeoLocation
  lane?: number
}

// API response types
interface CameraResponse {
  id: string
  name: string
  rtsp_url: string
  hls_url: string
  location: {
    lat: number
    lng: number
    address?: string
  }
  type: 'smart' | 'standard'
  status: 'online' | 'offline' | 'error'
  lane?: number
}

// Transform API response to Camera type
function transformCamera(data: CameraResponse): Camera {
  return {
    id: data.id,
    name: data.name,
    rtspUrl: data.rtsp_url,
    hlsUrl: data.hls_url,
    location: data.location,
    type: data.type,
    status: data.status,
    lane: data.lane,
  }
}

// Transform Camera to API request format
function transformToRequest(data: CameraFormData): Partial<CameraResponse> {
  return {
    name: data.name,
    rtsp_url: data.rtspUrl,
    type: data.type,
    location: data.location,
    lane: data.lane,
  }
}

export const cameraService = {
  /**
   * Get all cameras
   */
  async getAll(): Promise<Camera[]> {
    const response = await apiClient.get<CameraResponse[]>('/cameras')
    // Backend'den gelen veri array değilse boş array döndür
    const data = Array.isArray(response.data) ? response.data : []
    return data.map(transformCamera)
  },

  /**
   * Get single camera by ID
   */
  async getById(id: string): Promise<Camera> {
    const response = await apiClient.get<CameraResponse>(`/cameras/${id}`)
    return transformCamera(response.data)
  },

  /**
   * Create new camera
   */
  async create(data: CameraFormData): Promise<Camera> {
    const response = await apiClient.post<CameraResponse>('/cameras', transformToRequest(data))
    return transformCamera(response.data)
  },

  /**
   * Update existing camera
   */
  async update(id: string, data: Partial<CameraFormData>): Promise<Camera> {
    const requestData: Partial<CameraResponse> = {}

    if (data.name !== undefined) requestData.name = data.name
    if (data.rtspUrl !== undefined) requestData.rtsp_url = data.rtspUrl
    if (data.type !== undefined) requestData.type = data.type
    if (data.location !== undefined) requestData.location = data.location
    if (data.lane !== undefined) requestData.lane = data.lane

    const response = await apiClient.patch<CameraResponse>(`/cameras/${id}`, requestData)
    return transformCamera(response.data)
  },

  /**
   * Delete camera
   */
  async delete(id: string): Promise<void> {
    await apiClient.delete(`/cameras/${id}`)
  },

  /**
   * Update camera status
   */
  async updateStatus(id: string, status: Camera['status']): Promise<Camera> {
    const response = await apiClient.patch<CameraResponse>(`/cameras/${id}/status`, { status })
    return transformCamera(response.data)
  },

  /**
   * Get camera stream URL (HLS)
   */
  async getStreamUrl(id: string): Promise<string> {
    const response = await apiClient.get<{ hls_url: string }>(`/cameras/${id}/stream`)
    return response.data.hls_url
  },

  /**
   * Test camera RTSP connection
   */
  async testConnection(rtspUrl: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>('/cameras/test-connection', { rtsp_url: rtspUrl })
    return response.data
  },

  /**
   * Get smart cameras only
   */
  async getSmartCameras(): Promise<Camera[]> {
    const response = await apiClient.get<CameraResponse[]>('/cameras?type=smart')
    return response.data.map(transformCamera)
  }
}
