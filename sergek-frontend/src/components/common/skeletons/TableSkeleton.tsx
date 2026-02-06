/**
 * Table Skeleton
 * Loading placeholder for data tables
 * Requirements: 8.3 - Table skeleton
 */

import { Skeleton } from './Skeleton'

interface TableRowSkeletonProps {
  columns?: number
  showCheckbox?: boolean
  showActions?: boolean
}

export function TableRowSkeleton({
  columns = 5,
  showCheckbox = false,
  showActions = true,
}: TableRowSkeletonProps) {
  return (
    <tr className="border-b border-neutral-100">
      {showCheckbox && (
        <td className="px-4 py-3">
          <Skeleton className="w-4 h-4 rounded" />
        </td>
      )}
      {Array.from({ length: columns }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <Skeleton className={`h-4 ${i === 0 ? 'w-32' : i === columns - 1 ? 'w-20' : 'w-24'}`} />
        </td>
      ))}
      {showActions && (
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            <Skeleton className="w-8 h-8 rounded-lg" />
            <Skeleton className="w-8 h-8 rounded-lg" />
          </div>
        </td>
      )}
    </tr>
  )
}

interface TableSkeletonProps {
  rows?: number
  columns?: number
  showHeader?: boolean
  showCheckbox?: boolean
  showActions?: boolean
}

export function TableSkeleton({
  rows = 5,
  columns = 5,
  showHeader = true,
  showCheckbox = false,
  showActions = true,
}: TableSkeletonProps) {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 overflow-hidden">
      <table className="w-full">
        {showHeader && (
          <thead className="bg-neutral-50 border-b border-neutral-200">
            <tr>
              {showCheckbox && (
                <th className="px-4 py-3 text-left">
                  <Skeleton className="w-4 h-4 rounded" />
                </th>
              )}
              {Array.from({ length: columns }).map((_, i) => (
                <th key={i} className="px-4 py-3 text-left">
                  <Skeleton className="h-4 w-20" />
                </th>
              ))}
              {showActions && (
                <th className="px-4 py-3 text-left">
                  <Skeleton className="h-4 w-16" />
                </th>
              )}
            </tr>
          </thead>
        )}
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <TableRowSkeleton
              key={i}
              columns={columns}
              showCheckbox={showCheckbox}
              showActions={showActions}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Compact list skeleton (for sidebar lists)
interface ListSkeletonProps {
  items?: number
  showIcon?: boolean
}

export function ListSkeleton({ items = 5, showIcon = true }: ListSkeletonProps) {
  return (
    <div className="space-y-2">
      {Array.from({ length: items }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-neutral-50">
          {showIcon && <Skeleton className="w-10 h-10 rounded-lg" />}
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  )
}
