/**
 * API Client Configuration
 * Axios instance with interceptors for error handling and logging
 * Requirements: 1.5
 */

import axios from 'axios'
import type { AxiosInstance, AxiosError, InternalAxiosRequestConfig, AxiosResponse } from 'axios'

// API Base URL - configurable via environment
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

// Custom error class for API errors
export class APIError extends Error {
  statusCode: number
  endpoint: string
  originalError?: AxiosError

  constructor(statusCode: number, endpoint: string, message: string, originalError?: AxiosError) {
    super(message)
    this.name = 'APIError'
    this.statusCode = statusCode
    this.endpoint = endpoint
    this.originalError = originalError
  }
}

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const timestamp = new Date().toISOString()
    console.log(`[${timestamp}] API Request: ${config.method?.toUpperCase()} ${config.url}`)

    if (config.params) {
      console.log('  Params:', config.params)
    }

    return config
  },
  (error: AxiosError) => {
    console.error('[API] Request Error:', error.message)
    return Promise.reject(error)
  }
)

// Response interceptor for logging and error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    const timestamp = new Date().toISOString()
    console.log(`[${timestamp}] API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error: AxiosError) => {
    const timestamp = new Date().toISOString()
    const endpoint = error.config?.url || 'unknown'
    const statusCode = error.response?.status || 0

    console.error(`[${timestamp}] API Error: ${statusCode} ${endpoint}`)
    console.error('  Message:', error.message)

    // Transform to custom APIError
    const apiError = new APIError(statusCode, endpoint, getErrorMessage(error), error)

    return Promise.reject(apiError)
  }
)

// Helper to extract meaningful error message
function getErrorMessage(error: AxiosError): string {
  if (error.response) {
    // Server responded with error status
    const data = error.response.data as { message?: string; error?: string }
    return data?.message || data?.error || `Server error: ${error.response.status}`
  } else if (error.request) {
    // Request made but no response
    return 'Network error: No response from server'
  } else {
    // Request setup error
    return error.message || 'Unknown error occurred'
  }
}

export { apiClient }
export default apiClient
