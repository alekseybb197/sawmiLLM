"""
Модуль для восстановления и поиска строк в логах.
Содержит функции для сопоставления строк из prepared файлов с cleaned файлами,
а также общие утилиты для загрузки и декодирования Drain-последовательностей.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, cast
from pathlib import Path
from difflib import SequenceMatcher

from modules.config import Config


# === Утилиты для загрузки данных ===

def load_jsonl(path: Path) -> List[Dict[Any, Any]]:
    """
    Загружает JSONL файл (каждая строка - отдельный JSON объект).
    
    Args:
        path: Путь к JSONL файлу
        
    Returns:
        Список словарей, каждый словарь - одна запись из файла
    """
    with path.open("r", encoding="utf-8") as f:
        return [cast(Dict[Any, Any], json.loads(line)) for line in f if line.strip()]


def load_json(path: Path) -> Dict[Any, Any]:
    """
    Загружает JSON файл (один объект).
    
    Args:
        path: Путь к JSON файлу
        
    Returns:
        Словарь с данными из JSON файла
    """
    with path.open("r", encoding="utf-8") as f:
        return cast(Dict[Any, Any], json.load(f))


def decode_sequence(seq: List[str], templates: Dict[str, str]) -> List[str]:
    """
    Преобразует последовательность event_id в читаемые шаблоны логов.
    
    Args:
        seq: Список event_id (например, ["E1", "E2", "E3"])
        templates: Словарь шаблонов {event_id: template_string}
        
    Returns:
        Список шаблонов, соответствующих event_id из seq.
        Если event_id не найден в templates, возвращается "[UNKNOWN_EVENT_ID: EXXX]"
        
    Пример:
        templates = {"E1": "Agent name: <*>", "E2": "Agent machine name: <*>"}
        decode_sequence(["E1", "E2", "E999"], templates)
        -> ["Agent name: <*>", "Agent machine name: <*>", "[UNKNOWN_EVENT_ID: E999]"]
    """
    decoded = []
    for eid in seq:
        if eid in templates:
            decoded.append(templates[eid])
        else:
            decoded.append(f"[UNKNOWN_EVENT_ID: {eid}]")
    return decoded


# === Функции для поиска и сопоставления строк ===


def extract_content_from_recovery_line(recovery_line: str) -> str:
    """
    Извлекает основное содержимое из строки recovery, убирая номер и timestamp.
    
    Args:
        recovery_line: Строка вида "001. [timestamp] content" или "001. content"
        
    Returns:
        Очищенное содержимое строки
    """
    # Убираем номер в начале (001., 002., и т.д.)
    content = re.sub(r'^\d{3}\.\s*', '', recovery_line)
    
    # Убираем timestamp в квадратных скобках [timestamp]
    content = re.sub(r'^\[[^\]]+\]\s*', '', content)
    
    # Убираем префикс "[нет оригинала]"
    content = re.sub(r'^\[нет оригинала\]\s*', '', content)
    
    return content.strip()


def normalize_line(line: str) -> str:
    """
    Нормализует строку для сравнения: убирает лишние пробелы, приводит к нижнему регистру.
    
    Args:
        line: Исходная строка
        
    Returns:
        Нормализованная строка
    """
    # Убираем лишние пробелы и приводим к нижнему регистру
    normalized = re.sub(r'\s+', ' ', line.strip().lower())
    return normalized


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Вычисляет схожесть между двумя строками используя SequenceMatcher.
    
    Args:
        str1: Первая строка
        str2: Вторая строка
        
    Returns:
        Коэффициент схожести от 0.0 до 1.0
    """
    return SequenceMatcher(None, str1, str2).ratio()


def find_matching_line(
    recovery_content: str, 
    prepared_lines: List[str], 
    cursor: int,
    debug_stats: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[int, str]]:
    """
    Ищет совпадающую строку в prepared_lines начиная с позиции cursor.
    
    Args:
        recovery_content: Содержимое строки из recovery (уже очищенное от префиксов)
        prepared_lines: Список строк из prepared файла
        cursor: Текущая позиция курсора в prepared_lines
        debug_stats: Опциональный словарь для сбора статистики отладки
        
    Returns:
        Кортеж (индекс_найденной_строки, строка) или None, если совпадение не найдено
    """
    if not recovery_content or cursor >= len(prepared_lines):
        if debug_stats is not None:
            debug_stats["skipped_cursor_out_of_range"] = debug_stats.get("skipped_cursor_out_of_range", 0) + 1
        return None
    
    # Очищаем recovery_content (только strip, без нормализации для точного совпадения)
    recovery_stripped = recovery_content.strip()
    
    if not recovery_stripped:
        if debug_stats is not None:
            debug_stats["skipped_empty_recovery"] = debug_stats.get("skipped_empty_recovery", 0) + 1
        return None
    
    # Нормализуем для сравнения по схожести (для случаев с вариациями)
    recovery_normalized = normalize_line(recovery_stripped)
    
    best_match = None
    best_similarity = 0.0
    best_index = cursor
    
    # Ограничиваем область поиска: ищем в пределах разумного окна (cursor до cursor + 100)
    search_limit = min(cursor + 100, len(prepared_lines))
    
    exact_match_found = False
    exact_normalized_found = False
    checked_lines = 0
    
    # Ищем совпадение начиная с cursor
    for i in range(cursor, search_limit):
        prepared_line = prepared_lines[i]
        prepared_stripped = prepared_line.strip()
        
        if not prepared_stripped:
            continue
        
        checked_lines += 1
        
        # 1. ПРИОРИТЕТ: Точное совпадение без нормализации (сохраняет кавычки, регистр и т.д.)
        if recovery_stripped == prepared_stripped:
            exact_match_found = True
            if debug_stats is not None:
                debug_stats["exact_matches"] = debug_stats.get("exact_matches", 0) + 1
            return (i, prepared_line)
        
        # 2. Точное совпадение нормализованных строк (без учета регистра и пробелов)
        prepared_normalized = normalize_line(prepared_stripped)
        if recovery_normalized == prepared_normalized:
            exact_normalized_found = True
            # Если нашли точное нормализованное совпадение, но еще не нашли лучшее, сохраняем
            if best_match is None or best_similarity < 1.0:
                best_similarity = 1.0
                best_match = prepared_line
                best_index = i
                # Не прерываем сразу, продолжаем поиск точного совпадения без нормализации
                continue
        
        # 3. Высокая схожесть через SequenceMatcher
        similarity = calculate_similarity(recovery_normalized, prepared_normalized)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = prepared_line
            best_index = i
        
        # 4. Проверка наличия ключевых слов (если recovery содержит значимые слова)
        recovery_words = set(recovery_normalized.split())
        prepared_words = set(prepared_normalized.split())
        
        # Убираем служебные слова
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        recovery_words = {w for w in recovery_words if w not in stop_words and len(w) > 2}
        prepared_words = {w for w in prepared_words if w not in stop_words and len(w) > 2}
        
        # Если есть значительное пересечение ключевых слов (>=50%)
        if recovery_words and prepared_words:
            common_words = recovery_words & prepared_words
            word_overlap = len(common_words) / max(len(recovery_words), len(prepared_words))
            if word_overlap >= 0.5 and similarity > 0.5:
                # Если схожесть достаточно высока, используем это совпадение
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = prepared_line
                    best_index = i
    
    if debug_stats is not None:
        debug_stats["checked_lines"] = debug_stats.get("checked_lines", 0) + checked_lines
        if exact_normalized_found:
            debug_stats["exact_normalized_matches"] = debug_stats.get("exact_normalized_matches", 0) + 1
    
    # Если нашли точное совпадение (нормализованное), возвращаем его сразу
    if exact_normalized_found and best_match:
        return (best_index, best_match)
    
    # Возвращаем лучшее совпадение, если схожесть достаточно высока
    if best_match and best_similarity >= 0.5:
        if debug_stats is not None:
            debug_stats["similarity_matches"] = debug_stats.get("similarity_matches", 0) + 1
            debug_stats["avg_similarity"] = debug_stats.get("avg_similarity", 0.0) * (debug_stats.get("similarity_matches", 0) - 1) + best_similarity
            if debug_stats.get("similarity_matches", 0) > 0:
                debug_stats["avg_similarity"] = debug_stats["avg_similarity"] / debug_stats["similarity_matches"]
        return (best_index, best_match)
    
    if debug_stats is not None:
        debug_stats["not_found"] = debug_stats.get("not_found", 0) + 1
    
    return None


def check_recovery_pattern(prepared_line: str, config: Config) -> Optional[Tuple[str, str]]:
    """
    Проверяет, соответствует ли строка из prepared паттерну recovery: из dataset.filter_patterns.
    
    Args:
        prepared_line: Строка из prepared файла
        config: Объект Config из modules.config
        
    Returns:
        Кортеж (pattern_name, pattern_str) если найден recovery паттерн, иначе None.
        pattern_str - это regexp pattern для поиска в cleaned файле.
    """
    filter_patterns = config.get("dataset.filter_patterns", {})
    
    # Проходим по всем паттернам в порядке их указания, у которых есть recovery
    for pattern_name, pattern_config in filter_patterns.items():
        recovery_pattern = pattern_config.get("recovery")
        pattern_str = pattern_config.get("pattern")
        
        if recovery_pattern and pattern_str:
            # Компилируем паттерн recovery как регексп для проверки
            try:
                recovery_re = re.compile(recovery_pattern)
                # Используем search вместо match, чтобы искать подстроку в любой части строки
                if recovery_re.search(prepared_line):
                    # Если строка содержит recovery паттерн, возвращаем pattern для поиска
                    return (pattern_name, pattern_str)
            except re.error:
                # Если паттерн невалидный, пропускаем
                continue
    
    # Если ни один recovery паттерн не подошел, возвращаем None
    return None


def find_matching_line_by_pattern(
    pattern_str: str,
    cleaned_lines: List[str],
    cursor: int,
    debug_stats: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[int, str]]:
    """
    Ищет строку в cleaned_lines, соответствующую паттерну pattern_str, начиная с позиции cursor.
    
    Args:
        pattern_str: Регулярное выражение для поиска (pattern из filter_patterns)
        cleaned_lines: Список строк из cleaned файла
        cursor: Текущая позиция курсора в cleaned_lines
        debug_stats: Опциональный словарь для сбора статистики
        
    Returns:
        Кортеж (индекс_найденной_строки, строка) или None, если совпадение не найдено
    """
    if cursor >= len(cleaned_lines):
        return None
    
    # Компилируем паттерн для поиска
    try:
        pattern_re = re.compile(pattern_str)
    except re.error:
        # Если паттерн невалидный, возвращаем None
        if debug_stats is not None:
            debug_stats["pattern_compile_errors"] = debug_stats.get("pattern_compile_errors", 0) + 1
        return None
    
    # Ищем в пределах окна (cursor до cursor + 100)
    search_limit = min(cursor + 100, len(cleaned_lines))
    
    # Ищем первое совпадение начиная с cursor
    for i in range(cursor, search_limit):
        cleaned_line = cleaned_lines[i]
        
        # Проверяем, содержит ли строка паттерн (используем search для поиска подстроки)
        if pattern_re.search(cleaned_line):
            if debug_stats is not None:
                debug_stats["pattern_matches"] = debug_stats.get("pattern_matches", 0) + 1
            return (i, cleaned_line)
    
    # Если не найдено в окне, возвращаем None
    if debug_stats is not None:
        debug_stats["pattern_not_found"] = debug_stats.get("pattern_not_found", 0) + 1
    
    return None


def load_cleaned_file(cleaned_dir: Path, pipeline_id: str, build_id: str) -> Optional[List[str]]:
    """
    Загружает строки из исходного файла cleaned.
    
    Args:
        cleaned_dir: Директория с cleaned файлами
        pipeline_id: Идентификатор пайплайны
        build_id: Идентификатор прогона
        
    Returns:
        Список строк из файла или None, если файл не найден
    """
    cleaned_file = cleaned_dir / f"{pipeline_id}-{build_id}-logs_content.raw"
    if not cleaned_file.exists():
        return None
    
    with cleaned_file.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

