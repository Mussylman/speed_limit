/**
 * MapEditControls Component - Camera add/edit mode controls
 * Requirements: 4.4 (add camera), 4.5 (edit/delete), 4.6 (update location)
 */

import { useTranslation } from 'react-i18next'
import { useMapStore } from '../../stores/mapStore'
import { Edit3, Plus, X } from 'lucide-react'

interface MapEditControlsProps {
  onAddCamera?: () => void
  showAddButton?: boolean
}

export function MapEditControls({ onAddCamera, showAddButton = true }: MapEditControlsProps) {
  const { t } = useTranslation()
  const { isEditMode, toggleEditMode } = useMapStore()

  return (
    <div className="absolute top-6 right-6 z-[1001] flex flex-col gap-3">
      {/* Edit mode toggle */}
      <button
        onClick={toggleEditMode}
        className={`flex items-center gap-2 px-4 py-3 rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-xl transform hover:scale-105 ${
          isEditMode
            ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white'
            : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
        }`}
      >
        {isEditMode ? (
          <>
            <X size={16} />
            {t('map.exitEdit')}
          </>
        ) : (
          <>
            <Edit3 size={16} />
            {t('map.editMode')}
          </>
        )}
      </button>

      {/* Add camera button (only in edit mode) */}
      {isEditMode && showAddButton && (
        <button
          onClick={onAddCamera}
          className="flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white rounded-xl font-semibold text-sm transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          <Plus size={16} />
          {t('map.addCamera')}
        </button>
      )}

      {/* Instructions tooltip */}
      {isEditMode && (
        <div className="bg-white border border-gray-200 text-gray-700 text-sm px-4 py-3 rounded-xl max-w-[220px] shadow-lg">
          <p className="mb-2 font-medium">{t('map.instructions')}:</p>
          <p className="mb-1">• {t('map.clickToAdd')}</p>
          <p>• {t('map.dragToMove')}</p>
        </div>
      )}
    </div>
  )
}
