/**
 * PageTransition Component
 * Wraps page content with smooth enter/exit animations
 * Requirements: 8.5 - Smooth page transitions
 */

import { motion } from 'framer-motion'
import type { ReactNode } from 'react'
import { pageVariants } from './variants'

interface PageTransitionProps {
  children: ReactNode
  className?: string
}

export function PageTransition({ children, className = '' }: PageTransitionProps) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}
      className={className}
    >
      {children}
    </motion.div>
  )
}
