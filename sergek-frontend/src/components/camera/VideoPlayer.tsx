/**
 * VideoPlayer - RTSP/HLS Video Stream Player
 * Supports both RTSP and HLS streams with fallback
 */

import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Hls from 'hls.js'
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Maximize, 
  Loader2,
  Wifi,
  WifiOff
} from 'lucide-react'

interface VideoPlayerProps {
  rtspUrl?: string
  hlsUrl?: string
  poster?: string
  className?: string
  autoPlay?: boolean
  muted?: boolean
  controls?: boolean
  onError?: (error: string) => void
  onStatusChange?: (status: 'loading' | 'playing' | 'paused' | 'error') => void
}

export function VideoPlayer({
  rtspUrl,
  hlsUrl,
  poster,
  className = '',
  autoPlay = true,
  muted = true,
  controls = true,
  onError,
  onStatusChange
}: VideoPlayerProps) {
  const { t } = useTranslation()
  const videoRef = useRef<HTMLVideoElement>(null)
  const hlsRef = useRef<Hls | null>(null)
  
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(muted)
  const [volume, setVolume] = useState(0.5)
  const [status, setStatus] = useState<'loading' | 'playing' | 'paused' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)

  // Update status and notify parent
  const updateStatus = (newStatus: typeof status) => {
    setStatus(newStatus)
    onStatusChange?.(newStatus)
  }

  // Initialize HLS player
  useEffect(() => {
    const video = videoRef.current
    if (!video || !hlsUrl) return

    // Check if HLS is supported
    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        backBufferLength: 90,
        maxBufferLength: 30,
        maxMaxBufferLength: 60,
      })

      hlsRef.current = hls
      hls.loadSource(hlsUrl)
      hls.attachMedia(video)

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        updateStatus('paused')
        if (autoPlay) {
          video.play().catch(err => {
            console.error('Auto-play failed:', err)
            updateStatus('paused')
          })
        }
      })

      hls.on(Hls.Events.ERROR, (_, data) => {
        console.error('HLS Error:', data)
        const errorMsg = `Stream error: ${data.details || 'Unknown error'}`
        setError(errorMsg)
        updateStatus('error')
        onError?.(errorMsg)
      })

    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Native HLS support (Safari)
      video.src = hlsUrl
      video.addEventListener('loadedmetadata', () => {
        updateStatus('paused')
        if (autoPlay) {
          video.play().catch(err => {
            console.error('Auto-play failed:', err)
            updateStatus('paused')
          })
        }
      })
    } else {
      const errorMsg = 'HLS not supported in this browser'
      setError(errorMsg)
      updateStatus('error')
      onError?.(errorMsg)
    }

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }
    }
  }, [hlsUrl, autoPlay, onError])

  // Video event handlers
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handlePlay = () => {
      setIsPlaying(true)
      updateStatus('playing')
    }

    const handlePause = () => {
      setIsPlaying(false)
      updateStatus('paused')
    }

    const handleError = () => {
      const errorMsg = 'Video playback error'
      setError(errorMsg)
      updateStatus('error')
      onError?.(errorMsg)
    }

    const handleVolumeChange = () => {
      setVolume(video.volume)
      setIsMuted(video.muted)
    }

    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('error', handleError)
    video.addEventListener('volumechange', handleVolumeChange)

    return () => {
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('error', handleError)
      video.removeEventListener('volumechange', handleVolumeChange)
    }
  }, [onError])

  // Control handlers
  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      video.pause()
    } else {
      video.play().catch(err => {
        console.error('Play failed:', err)
      })
    }
  }

  const toggleMute = () => {
    const video = videoRef.current
    if (!video) return

    video.muted = !video.muted
  }

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current
    if (!video) return

    const newVolume = parseFloat(e.target.value)
    video.volume = newVolume
    setVolume(newVolume)
  }

  const toggleFullscreen = () => {
    const video = videoRef.current
    if (!video) return

    if (!document.fullscreenElement) {
      video.requestFullscreen().catch(err => {
        console.error('Fullscreen failed:', err)
      })
    } else {
      document.exitFullscreen()
    }
  }

  // Render status overlay
  const renderStatusOverlay = () => {
    if (status === 'loading') {
      return (
        <div className="absolute inset-0 flex items-center justify-center bg-neutral-900/80 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 text-white">
            <Loader2 className="w-8 h-8 animate-spin" />
            <p className="text-sm">{t('camera.connecting')}</p>
          </div>
        </div>
      )
    }

    if (status === 'error') {
      return (
        <div className="absolute inset-0 flex items-center justify-center bg-neutral-900/90 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 text-white text-center p-6">
            <WifiOff className="w-12 h-12 text-error-400" />
            <div>
              <p className="font-medium mb-1">{t('camera.streamError')}</p>
              <p className="text-sm text-neutral-300">{error}</p>
            </div>
            <button 
              onClick={() => window.location.reload()} 
              className="btn btn--secondary btn--sm mt-2"
            >
              {t('common.retry')}
            </button>
          </div>
        </div>
      )
    }

    return null
  }

  return (
    <div className={`relative bg-neutral-900 rounded-xl overflow-hidden ${className}`}>
      {/* Video Element */}
      <video
        ref={videoRef}
        className="w-full h-full object-cover"
        poster={poster}
        muted={isMuted}
        playsInline
        preload="metadata"
      />

      {/* Status Overlay */}
      {renderStatusOverlay()}

      {/* Stream Status Indicator */}
      <div className="absolute top-4 left-4">
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
          status === 'playing' 
            ? 'bg-success-500/90 text-white' 
            : status === 'error'
            ? 'bg-error-500/90 text-white'
            : 'bg-neutral-800/90 text-neutral-300'
        }`}>
          {status === 'playing' ? (
            <>
              <Wifi className="w-3 h-3" />
              <span>{t('camera.live')}</span>
            </>
          ) : status === 'error' ? (
            <>
              <WifiOff className="w-3 h-3" />
              <span>{t('camera.offline')}</span>
            </>
          ) : (
            <>
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>{t('camera.connecting')}</span>
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      {controls && status !== 'error' && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-neutral-900/90 to-transparent p-4">
          <div className="flex items-center gap-3">
            {/* Play/Pause */}
            <button
              onClick={togglePlay}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              disabled={status === 'loading'}
            >
              {isPlaying ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5" />
              )}
            </button>

            {/* Volume */}
            <div className="flex items-center gap-2">
              <button
                onClick={toggleMute}
                className="p-1.5 rounded text-white/80 hover:text-white transition-colors"
              >
                {isMuted ? (
                  <VolumeX className="w-4 h-4" />
                ) : (
                  <Volume2 className="w-4 h-4" />
                )}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={volume}
                onChange={handleVolumeChange}
                className="w-16 h-1 bg-white/20 rounded-full appearance-none slider"
              />
            </div>

            <div className="flex-1" />

            {/* Fullscreen */}
            <button
              onClick={toggleFullscreen}
              className="p-1.5 rounded text-white/80 hover:text-white transition-colors"
            >
              <Maximize className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* RTSP Info (if available) */}
      {rtspUrl && (
        <div className="absolute top-4 right-4">
          <div className="px-2 py-1 bg-neutral-800/90 text-neutral-300 text-xs rounded font-mono">
            RTSP
          </div>
        </div>
      )}
    </div>
  )
}