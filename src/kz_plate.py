# kz_plate.py
# KZ plate format: XXX YYY XX (3 digits + 3 letters + 2 digit region)
# Corrections, scoring, character voting utilities

from typing import Dict

# === Таблицы замен ===

# Буква → цифра (для позиций 0-2, 6-7)
CHAR_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0', 'U': '0',
    'I': '1', 'L': '1', 'J': '1',
    'Z': '2', 'B': '8', 'S': '5', 'G': '6', 'T': '7',
}

# Цифра → буква (для позиций 3-5)
CHAR_TO_LETTER = {
    '0': 'O', '1': 'I', '8': 'B', '9': 'B',
    '5': 'S', '6': 'G', '7': 'T', '2': 'Z',
}

# Визуально похожие буквы (OCR часто путает)
LETTER_CONFUSIONS = {
    'A': ['Z', 'H'],
    'C': ['G', 'O', 'Z'],
    'Z': ['A', 'C', '2'],
    'N': ['H', 'M'],
    'H': ['N', 'A'],
    'T': ['I', 'Y'],
    'E': ['F', 'B'],
}

# Направление OCR-ошибок: (misread, correct) → correct
OCR_PREFER = {
    ('A', 'Z'): 'Z',   # Z misread as A (diagonal lines)
    ('C', 'Z'): 'Z',   # Z misread as C (small resolution)
    ('H', 'N'): 'N',   # N misread as H
    ('N', 'H'): 'N',   # prefer N over H
    ('I', 'T'): 'T',   # T misread as I (cross-bar lost)
}

# Допустимые регионы KZ (01-21)
KZ_REGIONS = {f'{i:02d}' for i in range(1, 22)}


# === Позиционная коррекция ===

def fix_kz_8chars(chars: list) -> list:
    """Позиционная коррекция 8-символьного KZ номера (XXX YYY XX)."""
    for i in range(3):
        if chars[i].isalpha():
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
    for i in range(3, 6):
        if chars[i].isdigit():
            chars[i] = CHAR_TO_LETTER.get(chars[i], chars[i])
    for i in range(6, 8):
        if chars[i].isalpha():
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
    return chars


def fix_kz_7chars(chars: list) -> list:
    """Позиционная коррекция 7-символьного KZ (потеряна 1 буква: XXX YY XX)."""
    for i in range(3):
        if chars[i].isalpha():
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
    for i in range(3, 5):
        if chars[i].isdigit():
            chars[i] = CHAR_TO_LETTER.get(chars[i], chars[i])
    for i in range(5, 7):
        if chars[i].isalpha():
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
    return chars


# === Оценка соответствия формату ===

def kz_score(text: str) -> int:
    """Оценка соответствия 8-символьного KZ формату (0-8)."""
    if len(text) != 8:
        return 0
    score = 0
    for i in range(3):
        if text[i].isdigit():
            score += 1
    for i in range(3, 6):
        if text[i].isalpha():
            score += 1
    for i in range(6, 8):
        if text[i].isdigit():
            score += 1
    return score


def kz_score_7(text: str) -> int:
    """Оценка соответствия 7-символьного KZ формату (0-7)."""
    if len(text) != 7:
        return 0
    score = 0
    for i in range(3):
        if text[i].isdigit():
            score += 1
    for i in range(3, 5):
        if text[i].isalpha():
            score += 1
    for i in range(5, 7):
        if text[i].isdigit():
            score += 1
    return score


# === Основная коррекция ===

def fix_kz_plate(text: str) -> str:
    """Исправляет типичные ошибки OCR для казахстанских номеров.

    1. 8 символов — позиционная коррекция (O↔0, I↔1, B↔8, S↔5)
    2. 7 символов — пробуем вставить пропущенный символ
    3. 9 символов — пробуем удалить лишний символ
    """
    if not text or len(text) < 7 or len(text) > 9:
        return text

    if len(text) == 8:
        return ''.join(fix_kz_8chars(list(text)))

    if len(text) == 7:
        fixed7_str = ''.join(fix_kz_7chars(list(text)))

        # Пробуем вставить букву в позиции 3,4,5
        best = None
        best_score = -1
        for insert_pos in [3, 4, 5]:
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                candidate = text[:insert_pos] + letter + text[insert_pos:]
                fixed_str = ''.join(fix_kz_8chars(list(candidate)))
                score = kz_score(fixed_str)
                if fixed_str[6:8] in KZ_REGIONS:
                    score += 2
                if score > best_score:
                    best_score = score
                    best = fixed_str

        # Пробуем вставить цифру в позиции 0,1,2,6,7
        for insert_pos in [0, 1, 2, 6, 7]:
            for digit in '0123456789':
                candidate = text[:insert_pos] + digit + text[insert_pos:]
                fixed_str = ''.join(fix_kz_8chars(list(candidate)))
                score = kz_score(fixed_str)
                if fixed_str[6:8] in KZ_REGIONS:
                    score += 2
                if score > best_score:
                    best_score = score
                    best = fixed_str

        if best and best_score >= 9:
            return best
        return fixed7_str

    if len(text) == 9:
        best = None
        best_score = (-1, -1)
        for i in range(9):
            candidate = text[:i] + text[i+1:]
            native = kz_score(candidate)
            fixed_str = ''.join(fix_kz_8chars(list(candidate)))
            fmt = kz_score(fixed_str)
            if fixed_str[6:8] in KZ_REGIONS:
                fmt += 2
            score = (fmt, native)
            if score > best_score:
                best_score = score
                best = fixed_str
        if best and best_score[0] >= 9:
            return best

    return text


# === Голосование по символам ===

def merge_texts_charwise(texts: list) -> str:
    """Посимвольное мажоритарное голосование по нескольким OCR-чтениям.

    Группирует визуально похожие символы (A/Z/C) и суммирует их голоса.
    При равенстве — предпочитает первое (оригинальное) чтение.
    """
    if not texts:
        return ""
    length = len(texts[0])
    merged = []
    for i in range(length):
        counts: Dict[str, int] = {}
        for t in texts:
            if i < len(t):
                c = t[i]
                counts[c] = counts.get(c, 0) + 1

        # Группируем визуально похожие символы
        grouped: Dict[str, int] = {}
        used = set()
        for c, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if c in used:
                continue
            group_total = cnt
            used.add(c)
            for alt in LETTER_CONFUSIONS.get(c, []):
                if alt in counts and alt not in used:
                    group_total += counts[alt]
                    used.add(alt)
            grouped[c] = group_total

        best_char = texts[0][i] if i < len(texts[0]) else ''
        best_count = 0
        for c, cnt in grouped.items():
            if cnt > best_count:
                best_count = cnt
                best_char = c
        merged.append(best_char)
    return ''.join(merged)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Расстояние Левенштейна между двумя строками."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]
