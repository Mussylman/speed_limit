import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import ru from './locales/ru.json'
import kk from './locales/kk.json'
import en from './locales/en.json'

const resources = {
  ru: { translation: ru },
  kk: { translation: kk },
  en: { translation: en },
}

i18n.use(initReactI18next).init({
  resources,
  lng: 'ru', // Default language - Russian
  fallbackLng: 'ru', // Fallback to Russian
  interpolation: {
    escapeValue: false,
  },
  // Save language preference to localStorage
  detection: {
    order: ['localStorage', 'navigator'],
    caches: ['localStorage']
  }
})

export default i18n
