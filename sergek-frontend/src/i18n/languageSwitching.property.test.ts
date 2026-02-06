/**
 * Property Test: Language Switching Consistency
 * **Feature: sergek-camera-system, Property 8: Language Switching Consistency**
 * **Validates: Requirements 8.4**
 *
 * For any language switch operation, all visible UI text elements must update
 * to the selected language (Rusça, Kazakça, or İngilizce) without requiring page reload.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import i18n from './index'
import ru from './locales/ru.json'
import kk from './locales/kk.json'
import en from './locales/en.json'

const SUPPORTED_LANGUAGES = ['ru', 'kk', 'en'] as const
type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number]

const translations: Record<SupportedLanguage, typeof ru> = {
  ru,
  kk,
  en,
}

// Helper to get all translation keys recursively
function getAllTranslationKeys(obj: object, prefix = ''): string[] {
  const keys: string[] = []
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    if (typeof value === 'object' && value !== null) {
      keys.push(...getAllTranslationKeys(value, fullKey))
    } else {
      keys.push(fullKey)
    }
  }
  return keys
}

// Get nested value from object by dot-notation key
function getNestedValue(obj: object, key: string): string | undefined {
  const parts = key.split('.')
  let current: unknown = obj
  for (const part of parts) {
    if (current && typeof current === 'object' && part in current) {
      current = (current as Record<string, unknown>)[part]
    } else {
      return undefined
    }
  }
  return typeof current === 'string' ? current : undefined
}

describe('Property 8: Language Switching Consistency', () => {
  beforeEach(async () => {
    // Reset to default language before each test
    await i18n.changeLanguage('ru')
  })

  it('should switch language synchronously without page reload', async () => {
    const languageArbitrary = fc.constantFrom(...SUPPORTED_LANGUAGES)

    await fc.assert(
      fc.asyncProperty(languageArbitrary, async (targetLang) => {
        // Switch language
        await i18n.changeLanguage(targetLang)

        // Verify language changed
        expect(i18n.language).toBe(targetLang)

        // Verify translations are available
        const testKey = 'nav.dashboard'
        const translation = i18n.t(testKey)
        const expectedTranslation = getNestedValue(translations[targetLang], testKey)

        expect(translation).toBe(expectedTranslation)
      }),
      { numRuns: 100 }
    )
  })

  it('should have all translation keys present in all languages', () => {
    const baseKeys = getAllTranslationKeys(ru)

    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_LANGUAGES),
        fc.constantFrom(...baseKeys),
        (lang, key) => {
          const value = getNestedValue(translations[lang], key)
          // Every key in Russian should exist in all other languages
          expect(value).toBeDefined()
          expect(typeof value).toBe('string')
          expect(value!.length).toBeGreaterThan(0)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('should return correct translation for any key after language switch', async () => {
    const baseKeys = getAllTranslationKeys(ru)

    await fc.assert(
      fc.asyncProperty(
        fc.constantFrom(...SUPPORTED_LANGUAGES),
        fc.constantFrom(...baseKeys),
        async (lang, key) => {
          // Switch to target language
          await i18n.changeLanguage(lang)

          // Get translation via i18n
          const translation = i18n.t(key)

          // Get expected value directly from JSON
          const expected = getNestedValue(translations[lang], key)

          // They should match
          expect(translation).toBe(expected)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('should maintain language state across multiple switches', async () => {
    const languageSequenceArbitrary = fc.array(fc.constantFrom(...SUPPORTED_LANGUAGES), {
      minLength: 2,
      maxLength: 10,
    })

    await fc.assert(
      fc.asyncProperty(languageSequenceArbitrary, async (langSequence) => {
        for (const lang of langSequence) {
          await i18n.changeLanguage(lang)
          expect(i18n.language).toBe(lang)

          // Verify a sample translation is correct
          const translation = i18n.t('nav.dashboard')
          const expected = getNestedValue(translations[lang], 'nav.dashboard')
          expect(translation).toBe(expected)
        }
      }),
      { numRuns: 100 }
    )
  })

  it('should have unique translations for each language (no copy-paste)', () => {
    // For most keys, translations should differ between languages
    // (except for proper nouns, technical terms, etc.)
    const keysToCheck = [
      'nav.dashboard',
      'nav.cameras',
      'common.loading',
      'common.save',
      'common.cancel',
    ]

    for (const key of keysToCheck) {
      const ruValue = getNestedValue(ru, key)
      const kkValue = getNestedValue(kk, key)
      const enValue = getNestedValue(en, key)

      // At least 2 of 3 languages should have different translations
      const uniqueValues = new Set([ruValue, kkValue, enValue])
      expect(uniqueValues.size).toBeGreaterThanOrEqual(2)
    }
  })
})
