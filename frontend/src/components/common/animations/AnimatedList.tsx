/**
 * AnimatedList Components
 * Staggered list animations for items
 * Requirements: 8.5 - List item stagger animations
 */

import { motion } from 'framer-motion'
import type { ReactNode } from 'react'
import { listContainerVariants, listItemVariants, gridItemVariants, timing } from './variants'

interface AnimatedListProps {
  children: ReactNode
  className?: string
  staggerDelay?: number
}

export function AnimatedList({
  children,
  className = '',
  staggerDelay = timing.stagger,
}: AnimatedListProps) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={{
        ...listContainerVariants,
        animate: {
          ...listContainerVariants.animate,
          transition: {
            staggerChildren: staggerDelay,
            delayChildren: 0.1,
          },
        },
      }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

interface AnimatedListItemProps {
  children: ReactNode
  className?: string
  index?: number
  variant?: 'list' | 'grid'
}

export function AnimatedListItem({
  children,
  className = '',
  variant = 'list',
}: AnimatedListItemProps) {
  const variants = variant === 'grid' ? gridItemVariants : listItemVariants

  return (
    <motion.div variants={variants} className={className}>
      {children}
    </motion.div>
  )
}

// Grid-specific animated container
interface AnimatedGridProps {
  children: ReactNode
  className?: string
  staggerDelay?: number
}

export function AnimatedGrid({
  children,
  className = '',
  staggerDelay = timing.stagger,
}: AnimatedGridProps) {
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
            staggerChildren: staggerDelay,
            delayChildren: 0.05,
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

// Animated UL list
interface AnimatedULProps {
  children: ReactNode
  className?: string
  staggerDelay?: number
}

export function AnimatedUL({
  children,
  className = '',
  staggerDelay = timing.stagger,
}: AnimatedULProps) {
  return (
    <motion.ul
      initial="initial"
      animate="animate"
      exit="exit"
      variants={{
        initial: { opacity: 0 },
        animate: {
          opacity: 1,
          transition: {
            staggerChildren: staggerDelay,
            delayChildren: 0.1,
          },
        },
        exit: { opacity: 0 },
      }}
      className={className}
    >
      {children}
    </motion.ul>
  )
}

// Animated LI item
interface AnimatedLIProps {
  children: ReactNode
  className?: string
}

export function AnimatedLI({ children, className = '' }: AnimatedLIProps) {
  return (
    <motion.li variants={listItemVariants} className={className}>
      {children}
    </motion.li>
  )
}
