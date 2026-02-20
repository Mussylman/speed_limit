/**
 * Reusable animation variants for Framer Motion
 * Requirements: 8.5 - Smooth transitions and animations
 */

import type { Variants, Transition } from 'framer-motion'

// Timing presets - Arctic Command Center aesthetic: precise, controlled
export const timing = {
  fast: 0.15,
  normal: 0.25,
  slow: 0.4,
  stagger: 0.05,
} as const

// Easing presets
export const easing = {
  smooth: [0.4, 0, 0.2, 1],
  bounce: [0.68, -0.55, 0.265, 1.55],
  sharp: [0.4, 0, 0.6, 1],
  decelerate: [0, 0, 0.2, 1],
  accelerate: [0.4, 0, 1, 1],
} as const

// Default transition
export const defaultTransition: Transition = {
  duration: timing.normal,
  ease: easing.smooth,
}

// Page transition variants
export const pageVariants: Variants = {
  initial: {
    opacity: 0,
    y: 12,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: timing.normal,
      ease: easing.decelerate,
    },
  },
  exit: {
    opacity: 0,
    y: -8,
    transition: {
      duration: timing.fast,
      ease: easing.accelerate,
    },
  },
}

// Fade variants
export const fadeVariants: Variants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: { duration: timing.normal, ease: easing.smooth },
  },
  exit: {
    opacity: 0,
    transition: { duration: timing.fast, ease: easing.smooth },
  },
}

// Slide variants
export const slideUpVariants: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

export const slideDownVariants: Variants = {
  initial: { opacity: 0, y: -20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    y: 10,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

export const slideLeftVariants: Variants = {
  initial: { opacity: 0, x: 20 },
  animate: {
    opacity: 1,
    x: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    x: -10,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

export const slideRightVariants: Variants = {
  initial: { opacity: 0, x: -20 },
  animate: {
    opacity: 1,
    x: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    x: 10,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

// Scale variants
export const scaleVariants: Variants = {
  initial: { opacity: 0, scale: 0.95 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    scale: 0.98,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

// List container variants (for stagger children)
export const listContainerVariants: Variants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: {
      staggerChildren: timing.stagger,
      delayChildren: 0.1,
    },
  },
  exit: {
    opacity: 0,
    transition: {
      staggerChildren: timing.stagger / 2,
      staggerDirection: -1,
    },
  },
}

// List item variants
export const listItemVariants: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    y: -5,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

// Grid item variants (for camera grid)
export const gridItemVariants: Variants = {
  initial: { opacity: 0, scale: 0.9 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

// Modal/Overlay variants
export const overlayVariants: Variants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: { duration: timing.fast, ease: easing.smooth },
  },
  exit: {
    opacity: 0,
    transition: { duration: timing.fast, ease: easing.smooth },
  },
}

export const modalVariants: Variants = {
  initial: { opacity: 0, scale: 0.95, y: 10 },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: timing.normal, ease: easing.bounce },
  },
  exit: {
    opacity: 0,
    scale: 0.98,
    y: 5,
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

// Drawer variants
export const drawerRightVariants: Variants = {
  initial: { x: '100%' },
  animate: {
    x: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    x: '100%',
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

export const drawerLeftVariants: Variants = {
  initial: { x: '-100%' },
  animate: {
    x: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    x: '-100%',
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

export const drawerBottomVariants: Variants = {
  initial: { y: '100%' },
  animate: {
    y: 0,
    transition: { duration: timing.normal, ease: easing.decelerate },
  },
  exit: {
    y: '100%',
    transition: { duration: timing.fast, ease: easing.accelerate },
  },
}

// Card hover variants
export const cardHoverVariants: Variants = {
  initial: { scale: 1 },
  hover: {
    scale: 1.02,
    transition: { duration: timing.fast, ease: easing.smooth },
  },
  tap: {
    scale: 0.98,
    transition: { duration: timing.fast, ease: easing.smooth },
  },
}

// Pulse animation for status indicators
export const pulseVariants: Variants = {
  initial: { scale: 1, opacity: 1 },
  animate: {
    scale: [1, 1.2, 1],
    opacity: [1, 0.5, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'easeInOut',
    },
  },
}
