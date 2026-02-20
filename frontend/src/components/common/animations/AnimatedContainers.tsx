/**
 * Animated Container Components
 * Reusable animated wrappers for common animation patterns
 * Requirements: 8.5 - Smooth animations
 */

import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'
import {
  fadeVariants,
  slideUpVariants,
  slideDownVariants,
  slideLeftVariants,
  slideRightVariants,
  scaleVariants,
  timing,
  easing,
} from './variants'

interface AnimatedContainerProps {
  children: ReactNode
  className?: string
  delay?: number
  duration?: number
}

// Fade In animation
export function FadeIn({
  children,
  className = '',
  delay = 0,
  duration = timing.normal,
}: AnimatedContainerProps) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={{
        ...fadeVariants,
        animate: {
          ...fadeVariants.animate,
          transition: { duration, delay, ease: easing.smooth },
        },
      }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// Slide In animation with direction
type SlideDirection = 'up' | 'down' | 'left' | 'right'

interface SlideInProps extends AnimatedContainerProps {
  direction?: SlideDirection
}

const slideVariantsMap = {
  up: slideUpVariants,
  down: slideDownVariants,
  left: slideLeftVariants,
  right: slideRightVariants,
}

export function SlideIn({
  children,
  className = '',
  delay = 0,
  duration = timing.normal,
  direction = 'up',
}: SlideInProps) {
  const variants = slideVariantsMap[direction]

  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={{
        ...variants,
        animate: {
          ...variants.animate,
          transition: { duration, delay, ease: easing.decelerate },
        },
      }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// Scale In animation
export function ScaleIn({
  children,
  className = '',
  delay = 0,
  duration = timing.normal,
}: AnimatedContainerProps) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={{
        ...scaleVariants,
        animate: {
          ...scaleVariants.animate,
          transition: { duration, delay, ease: easing.decelerate },
        },
      }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// Stagger container for children animations
interface StaggerContainerProps extends AnimatedContainerProps {
  staggerDelay?: number
}

export function StaggerContainer({
  children,
  className = '',
  delay = 0,
  staggerDelay = timing.stagger,
}: StaggerContainerProps) {
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
            delayChildren: delay,
            staggerChildren: staggerDelay,
          },
        },
        exit: {
          opacity: 0,
          transition: {
            staggerChildren: staggerDelay / 2,
            staggerDirection: -1,
          },
        },
      }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// Stagger item (child of StaggerContainer)
interface StaggerItemProps {
  children: ReactNode
  className?: string
}

export function StaggerItem({ children, className = '' }: StaggerItemProps) {
  return (
    <motion.div
      variants={{
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
      }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// Motion div wrapper for custom animations
export type MotionDivProps = HTMLMotionProps<'div'>

export function MotionDiv(props: MotionDivProps) {
  return <motion.div {...props} />
}
