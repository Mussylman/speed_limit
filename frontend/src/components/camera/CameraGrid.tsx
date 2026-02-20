/**
 * CameraGrid Component
 * Displays cameras in grid or single view mode with responsive layout
 * Requirements: 1.1, 1.2, 1.4
 */

import { useTranslation } from 'react-i18next'
import { motion, AnimatePresence } from 'framer-motion'
import { Grid3X3, Maximize2, Camera as CameraIcon } from 'lucide-react'
import { CameraCard } from './CameraCard'
import { gridItemVariants, timing, easing } from '../common/animations/variants'
import type { Camera, CameraViewMode } from '../../types'

interface CameraGridProps {
  cameras: Camera[]
  viewMode: CameraViewMode
  selectedCameraId: string | null
  onCameraSelect: (id: string | null) => void
  onViewModeChange: (mode: CameraViewMode) => void
}

export function CameraGrid({
  cameras,
  viewMode,
  selectedCameraId,
  onCameraSelect,
  onViewModeChange,
}: CameraGridProps) {
  const { t } = useTranslation()

  const selectedCamera = selectedCameraId ? cameras.find((c) => c.id === selectedCameraId) : null

  const handleCameraClick = (cameraId: string) => {
    if (viewMode === 'single' && selectedCameraId === cameraId) {
      // Clicking selected camera in single mode - deselect and go back to grid
      onCameraSelect(null)
      onViewModeChange('grid')
    } else {
      onCameraSelect(cameraId)
      if (viewMode === 'grid') {
        // Double-click behavior: select in grid, click again to go single
      }
    }
  }

  const handleDoubleClick = (cameraId: string) => {
    onCameraSelect(cameraId)
    onViewModeChange('single')
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with view toggle */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            <CameraIcon className="w-5 h-5 text-primary-600" />
          </div>
          <div>
            <h2 className="heading-3">{t('pages.cameras.title')}</h2>
            <p className="text-sm text-neutral-500">
              {cameras.length} {cameras.length === 1 ? 'camera' : 'cameras'}
            </p>
          </div>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center gap-1 p-1 bg-neutral-100 rounded-lg">
          <button
            onClick={() => {
              onViewModeChange('grid')
              if (viewMode === 'single') onCameraSelect(null)
            }}
            className={`
              flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium
              transition-all duration-200
              ${
                viewMode === 'grid'
                  ? 'bg-white text-neutral-900 shadow-sm'
                  : 'text-neutral-500 hover:text-neutral-700'
              }
            `}
            aria-label="Grid view"
          >
            <Grid3X3 className="w-4 h-4" />
            <span className="hidden sm:inline">Grid</span>
          </button>
          <button
            onClick={() => {
              if (cameras.length > 0 && !selectedCameraId) {
                onCameraSelect(cameras[0].id)
              }
              onViewModeChange('single')
            }}
            className={`
              flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium
              transition-all duration-200
              ${
                viewMode === 'single'
                  ? 'bg-white text-neutral-900 shadow-sm'
                  : 'text-neutral-500 hover:text-neutral-700'
              }
            `}
            aria-label="Single view"
          >
            <Maximize2 className="w-4 h-4" />
            <span className="hidden sm:inline">Single</span>
          </button>
        </div>
      </div>

      {/* Camera display area */}
      {cameras.length === 0 ? (
        <EmptyState />
      ) : viewMode === 'single' && selectedCamera ? (
        <SingleView camera={selectedCamera} cameras={cameras} onCameraSelect={onCameraSelect} />
      ) : (
        <GridView
          cameras={cameras}
          selectedCameraId={selectedCameraId}
          onCameraClick={handleCameraClick}
          onCameraDoubleClick={handleDoubleClick}
        />
      )}
    </div>
  )
}

// Grid View Component
function GridView({
  cameras,
  selectedCameraId,
  onCameraClick,
  onCameraDoubleClick,
}: {
  cameras: Camera[]
  selectedCameraId: string | null
  onCameraClick: (id: string) => void
  onCameraDoubleClick: (id: string) => void
}) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={{
        initial: { opacity: 0 },
        animate: {
          opacity: 1,
          transition: {
            staggerChildren: timing.stagger,
            delayChildren: 0.05,
          },
        },
        exit: { opacity: 0 },
      }}
      className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4 auto-rows-fr"
    >
      <AnimatePresence mode="popLayout">
        {cameras.map((camera) => (
          <motion.div
            key={camera.id}
            variants={gridItemVariants}
            layout
            onDoubleClick={() => onCameraDoubleClick(camera.id)}
          >
            <CameraCard
              camera={camera}
              isSelected={selectedCameraId === camera.id}
              onClick={() => onCameraClick(camera.id)}
            />
          </motion.div>
        ))}
      </AnimatePresence>
    </motion.div>
  )
}

// Single View Component
function SingleView({
  camera,
  cameras,
  onCameraSelect,
}: {
  camera: Camera
  cameras: Camera[]
  onCameraSelect: (id: string) => void
}) {
  const { t } = useTranslation()

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: timing.normal, ease: easing.decelerate }}
      className="flex flex-col lg:flex-row gap-6 flex-1"
    >
      {/* Main camera view */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: timing.normal, delay: 0.1, ease: easing.decelerate }}
        className="flex-1 min-w-0"
      >
        <div className="h-full min-h-[400px] lg:min-h-[500px]">
          <CameraCard camera={camera} isSelected={true} showOverlay={true} />
        </div>
      </motion.div>

      {/* Camera selector sidebar */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: timing.normal, delay: 0.15, ease: easing.decelerate }}
        className="lg:w-72 xl:w-80 flex-shrink-0"
      >
        <div className="bg-neutral-50 rounded-xl p-4 h-full">
          <h3 className="text-sm font-semibold text-neutral-700 mb-3">
            {t('pages.cameras.title')}
          </h3>
          <div className="space-y-2 max-h-[400px] lg:max-h-[calc(100vh-300px)] overflow-y-auto pr-2">
            {cameras.map((cam, index) => (
              <motion.button
                key={cam.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: timing.normal,
                  delay: 0.2 + index * 0.03,
                  ease: easing.decelerate,
                }}
                onClick={() => onCameraSelect(cam.id)}
                className={`
                  w-full flex items-center gap-3 p-3 rounded-lg text-left
                  transition-all duration-200
                  ${
                    cam.id === camera.id
                      ? 'bg-primary-100 border-2 border-primary-500'
                      : 'bg-white border border-neutral-200 hover:border-primary-300 hover:bg-primary-50'
                  }
                `}
              >
                {/* Status dot */}
                <div className="relative flex-shrink-0">
                  <div
                    className={`
                    w-2.5 h-2.5 rounded-full
                    ${
                      cam.status === 'online'
                        ? 'bg-success-500'
                        : cam.status === 'error'
                          ? 'bg-error-500'
                          : 'bg-neutral-400'
                    }
                  `}
                  />
                  {cam.status === 'online' && (
                    <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-success-500 animate-ping opacity-75" />
                  )}
                </div>

                {/* Camera info */}
                <div className="flex-1 min-w-0">
                  <p
                    className={`
                    text-sm font-medium truncate
                    ${cam.id === camera.id ? 'text-primary-700' : 'text-neutral-700'}
                  `}
                  >
                    {cam.name}
                  </p>
                  <p className="text-xs text-neutral-500 truncate">
                    {cam.location.address ||
                      `${cam.location.lat.toFixed(4)}, ${cam.location.lng.toFixed(4)}`}
                  </p>
                </div>

                {/* Smart badge */}
                {cam.type === 'smart' && (
                  <span className="flex-shrink-0 text-[10px] font-bold text-primary-600 bg-primary-100 px-1.5 py-0.5 rounded">
                    SMART
                  </span>
                )}
              </motion.button>
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

// Empty State Component
function EmptyState() {
  const { t } = useTranslation()

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: timing.normal, ease: easing.decelerate }}
      className="flex-1 flex items-center justify-center"
    >
      <div className="text-center py-16">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: timing.normal, delay: 0.1, ease: easing.bounce }}
          className="w-16 h-16 mx-auto mb-4 bg-neutral-100 rounded-full flex items-center justify-center"
        >
          <CameraIcon className="w-8 h-8 text-neutral-400" />
        </motion.div>
        <motion.h3
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: timing.normal, delay: 0.15, ease: easing.decelerate }}
          className="text-lg font-semibold text-neutral-700 mb-2"
        >
          {t('common.noData')}
        </motion.h3>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: timing.normal, delay: 0.2, ease: easing.decelerate }}
          className="text-sm text-neutral-500 max-w-sm"
        >
          No cameras have been configured yet. Add cameras from the admin panel.
        </motion.p>
      </div>
    </motion.div>
  )
}
