/**
 * Animated Overlay Components
 * Modal and Drawer with smooth animations
 * Requirements: 8.5 - Modal/drawer animations
 */

import { motion, AnimatePresence } from 'framer-motion'
import type { ReactNode, MouseEvent } from 'react'
import { useEffect, useCallback } from 'react'
import { X } from 'lucide-react'
import {
  overlayVariants,
  modalVariants,
  drawerRightVariants,
  drawerLeftVariants,
  drawerBottomVariants,
} from './variants'

// Modal Component
interface ModalProps {
  isOpen: boolean
  onClose: () => void
  children: ReactNode
  title?: string
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  showCloseButton?: boolean
  closeOnOverlayClick?: boolean
  closeOnEscape?: boolean
}

const modalSizes = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-4xl',
}

export function Modal({
  isOpen,
  onClose,
  children,
  title,
  size = 'md',
  showCloseButton = true,
  closeOnOverlayClick = true,
  closeOnEscape = true,
}: ModalProps) {
  // Handle escape key
  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (closeOnEscape && e.key === 'Escape') {
        onClose()
      }
    },
    [closeOnEscape, onClose]
  )

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    }
    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = ''
    }
  }, [isOpen, handleEscape])

  const handleOverlayClick = (e: MouseEvent) => {
    if (closeOnOverlayClick && e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial="initial"
            animate="animate"
            exit="exit"
            variants={overlayVariants}
            className="absolute inset-0 bg-neutral-900/60 backdrop-blur-sm"
            onClick={handleOverlayClick}
          />

          {/* Modal content */}
          <motion.div
            initial="initial"
            animate="animate"
            exit="exit"
            variants={modalVariants}
            className={`
              relative w-full ${modalSizes[size]}
              bg-white rounded-2xl shadow-2xl
              max-h-[90vh] overflow-hidden
            `}
          >
            {/* Header */}
            {(title || showCloseButton) && (
              <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-100">
                {title && (
                  <h2 className="text-lg font-semibold text-neutral-900 font-display">{title}</h2>
                )}
                {showCloseButton && (
                  <button
                    onClick={onClose}
                    className="p-2 -mr-2 rounded-lg text-neutral-400 hover:text-neutral-600 hover:bg-neutral-100 transition-colors"
                    aria-label="Close modal"
                  >
                    <X className="w-5 h-5" />
                  </button>
                )}
              </div>
            )}

            {/* Body */}
            <div className="overflow-y-auto max-h-[calc(90vh-80px)]">{children}</div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}

// Drawer Component
type DrawerPosition = 'left' | 'right' | 'bottom'

interface DrawerProps {
  isOpen: boolean
  onClose: () => void
  children: ReactNode
  title?: string
  position?: DrawerPosition
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showCloseButton?: boolean
  closeOnOverlayClick?: boolean
  closeOnEscape?: boolean
}

const drawerVariantsMap = {
  left: drawerLeftVariants,
  right: drawerRightVariants,
  bottom: drawerBottomVariants,
}

const drawerSizes = {
  sm: 'w-80',
  md: 'w-96',
  lg: 'w-[28rem]',
  xl: 'w-[32rem]',
}

const drawerBottomSizes = {
  sm: 'h-1/4',
  md: 'h-1/3',
  lg: 'h-1/2',
  xl: 'h-2/3',
}

export function Drawer({
  isOpen,
  onClose,
  children,
  title,
  position = 'right',
  size = 'md',
  showCloseButton = true,
  closeOnOverlayClick = true,
  closeOnEscape = true,
}: DrawerProps) {
  // Handle escape key
  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (closeOnEscape && e.key === 'Escape') {
        onClose()
      }
    },
    [closeOnEscape, onClose]
  )

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    }
    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = ''
    }
  }, [isOpen, handleEscape])

  const handleOverlayClick = (e: MouseEvent) => {
    if (closeOnOverlayClick && e.target === e.currentTarget) {
      onClose()
    }
  }

  const variants = drawerVariantsMap[position]
  const isHorizontal = position === 'left' || position === 'right'

  const positionClasses = {
    left: 'left-0 top-0 bottom-0',
    right: 'right-0 top-0 bottom-0',
    bottom: 'bottom-0 left-0 right-0',
  }

  const sizeClass = isHorizontal ? drawerSizes[size] : drawerBottomSizes[size]

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50">
          {/* Backdrop */}
          <motion.div
            initial="initial"
            animate="animate"
            exit="exit"
            variants={overlayVariants}
            className="absolute inset-0 bg-neutral-900/60 backdrop-blur-sm"
            onClick={handleOverlayClick}
          />

          {/* Drawer content */}
          <motion.div
            initial="initial"
            animate="animate"
            exit="exit"
            variants={variants}
            className={`
              absolute ${positionClasses[position]} ${sizeClass}
              bg-white shadow-2xl
              ${isHorizontal ? 'h-full' : 'w-full rounded-t-2xl'}
              flex flex-col
            `}
          >
            {/* Header */}
            {(title || showCloseButton) && (
              <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-100 flex-shrink-0">
                {title && (
                  <h2 className="text-lg font-semibold text-neutral-900 font-display">{title}</h2>
                )}
                {showCloseButton && (
                  <button
                    onClick={onClose}
                    className="p-2 -mr-2 rounded-lg text-neutral-400 hover:text-neutral-600 hover:bg-neutral-100 transition-colors"
                    aria-label="Close drawer"
                  >
                    <X className="w-5 h-5" />
                  </button>
                )}
              </div>
            )}

            {/* Body */}
            <div className="flex-1 overflow-y-auto">{children}</div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}
