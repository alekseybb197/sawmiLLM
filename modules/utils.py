"""
Модуль с общими утилитами для обработки данных и работы с файлами.
Содержит функции для валидации, очистки данных и работы с прогресс-барами.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterable
from tqdm import tqdm
import json
from datetime import datetime
from modules.config import get_config

# Кэш для хранения скомпилированных регулярных выражений
_compiled_patterns_cache: Dict[str, List[Tuple[str, re.Pattern, str]]] = {}


def get_patterns_cache_info() -> Dict[str, int]:
    """
    Возвращает информацию о текущем состоянии кэша паттернов.
    
    Returns:
        Словарь с количеством закэшированных паттернов для каждого типа
    """
    return {
        pattern_type: len(patterns) 
        for pattern_type, patterns in _compiled_patterns_cache.items()
    }


def clear_patterns_cache(pattern_type: Optional[str] = None) -> None:
    """
    Очищает кэш скомпилированных паттернов.
    
    ВНИМАНИЕ: Очистка кэша приведет к повторной компиляции паттернов
    при следующем вызове compile_patterns. Обычно используется только при
    изменении конфигурации во время выполнения.
    
    Args:
        pattern_type: Тип паттернов для очистки (если None, очищает весь кэш)
    """
    if pattern_type is None:
        _compiled_patterns_cache.clear()
    elif pattern_type in _compiled_patterns_cache:
        del _compiled_patterns_cache[pattern_type]


def compile_patterns(pattern_type: str = "filter_patterns", verbose: bool = False) -> List[Tuple[str, re.Pattern, str]]:
    """
    Компилирует регулярные выражения из конфигурации и кэширует результат.
    
    Кэш сохраняется для всей сессии обработки и используется для всех файлов,
    что значительно ускоряет обработку при пакетной обработке.
    
    Args:
        pattern_type: Тип паттернов ("filter_patterns" или "clean_patterns")
        verbose: Выводить ли информацию о кэшировании (для отладки)
        
    Returns:
        Список кортежей (filter_name, compiled_pattern, replace_value)
    """
    # Проверяем кэш
    if pattern_type in _compiled_patterns_cache:
        if verbose:
            print(f"✓ Используются закэшированные паттерны для '{pattern_type}' "
                  f"({len(_compiled_patterns_cache[pattern_type])} паттернов)")
        return _compiled_patterns_cache[pattern_type]
    
    # Первая загрузка - компилируем паттерны
    if verbose:
        print(f"→ Компиляция паттернов '{pattern_type}'...")
    
    config = get_config()
    filters = config.get(f"dataset.{pattern_type}", {})
    
    if not filters:
        raise ValueError(
            f"Паттерны '{pattern_type}' не найдены в конфигурации. "
            f"Проверьте файл config.yaml в секции dataset.{pattern_type}"
        )
    
    compiled_patterns = []
    for filter_name, filter_config in filters.items():
        pattern_str = filter_config.get("pattern")
        replace_value = filter_config.get("replace", "")
        
        if not pattern_str:
            raise ValueError(
                f"Паттерн 'pattern' не найден в конфигурации {filter_name}. "
                f"Проверьте файл config.yaml в секции dataset.{pattern_type}.{filter_name}"
            )
        
        # Компилируем регулярное выражение
        try:
            compiled_pattern = re.compile(pattern_str)
            compiled_patterns.append((filter_name, compiled_pattern, replace_value))
        except re.error as e:
            raise ValueError(
                f"Ошибка компиляции регулярного выражения для {filter_name}: {e}. "
                f"Паттерн: {pattern_str}"
            ) from e
    
    # Сохраняем в кэш для использования во всех последующих вызовах
    _compiled_patterns_cache[pattern_type] = compiled_patterns
    
    if verbose:
        print(f"✓ Паттерны '{pattern_type}' скомпилированы и закэшированы "
              f"({len(compiled_patterns)} паттернов)")
    
    return compiled_patterns


def apply_filters(line: str, pattern_type: str = "filter_patterns", 
                  compiled_patterns: Optional[List[Tuple[str, re.Pattern, str]]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Применяет все фильтры к строке согласно конфигурации.
    Безусловно применяет все паттерны из указанного типа.
    Использует предкомпилированные регулярные выражения для оптимизации.
    
    Args:
        line: Исходная строка
        pattern_type: Тип паттернов ("filter_patterns" или "clean_patterns")
        compiled_patterns: Список предкомпилированных паттернов (если None, загружает из кэша)
        
    Returns:
        Tuple с обработанной строкой и статистикой фильтрации
    """
    # Используем переданные паттерны или загружаем из кэша
    if compiled_patterns is None:
        compiled_patterns = compile_patterns(pattern_type)
    
    processed_line = line
    filter_stats: Dict[str, Any] = {
        "original_length": len(line),
        "applied_filters": [],
        "replaced_by_filter": {}
    }
    
    # Применяем все скомпилированные паттерны
    for filter_name, compiled_pattern, replace_value in compiled_patterns:
        # БЕЗУСЛОВНО применяем замену ко всем строкам
        original_processed = processed_line
        processed_line = compiled_pattern.sub(replace_value, processed_line).strip()
        
        # Записываем статистику только если произошли изменения
        if original_processed != processed_line:
            applied_filters = filter_stats.get("applied_filters", [])
            assert isinstance(applied_filters, list)
            applied_filters.append(filter_name)
            replaced_by_filter = filter_stats.get("replaced_by_filter", {})
            assert isinstance(replaced_by_filter, dict)
            replaced_by_filter[filter_name] = {
                "replaced_with": replace_value,
                "was_removed": replace_value == ""
            }
    
    filter_stats["final_length"] = len(processed_line)
    return processed_line, filter_stats


def process_log_file(input_path: str, output_path: str, 
                    progress_callback: Optional[Callable[[int, int], None]] = None,
                    pattern_type: str = "filter_patterns",
                    remove_duplicates: bool = False) -> Dict[str, Any]:
    """
    Обрабатывает файл логов, применяя все фильтры согласно конфигурации.
    Использует предкомпилированные регулярные выражения для оптимизации.
    
    Args:
        input_path: Путь к входному файлу
        output_path: Путь к выходному файлу
        progress_callback: Функция обратного вызова для прогресса
        pattern_type: Тип паттернов ("filter_patterns" или "clean_patterns")
        remove_duplicates: Удалять ли повторяющиеся строки
        
    Returns:
        Словарь со статистикой обработки
    """
    config = get_config()
    encoding = config.get("dataset.encoding", "utf-8")
    remove_empty = config.get("dataset.remove_empty_lines", True)
    
    # Предкомпилируем паттерны один раз для всего файла
    # Кэш будет использоваться для всех последующих файлов при пакетной обработке
    compiled_patterns = compile_patterns(pattern_type)
    
    stats: Dict[str, Any] = {
        "total_lines": 0,
        "processed_lines": 0,
        "duplicate_lines": 0,
        "processing_time": 0.0,
        "filter_stats": {}
    }
    
    start_time = datetime.now()
    
    try:
        with open(input_path, 'r', encoding=encoding) as infile, \
             open(output_path, 'w', encoding=encoding) as outfile:
            
            # Получаем общее количество строк для прогресс-бара
            total_lines = sum(1 for _ in infile)
            infile.seek(0)
            
            with tqdm(total=total_lines, desc="Обработка логов") as pbar:
                # Переменная для хранения предыдущей строки (если включена проверка дубликатов)
                previous_line = None if remove_duplicates else None
                
                for line_num, line in enumerate(infile, 1):
                    total_lines_val = stats.get("total_lines", 0)
                    assert isinstance(total_lines_val, int)
                    stats["total_lines"] = total_lines_val + 1
                    
                    # Применяем все фильтры с использованием предкомпилированных паттернов
                    processed_line, filter_stats = apply_filters(line, pattern_type, compiled_patterns)
                    
                    # Обновляем статистику фильтров (универсально для всех фильтров)
                    applied_filters_list = filter_stats.get("applied_filters", [])
                    assert isinstance(applied_filters_list, list)
                    replaced_by_filter_dict = filter_stats.get("replaced_by_filter", {})
                    assert isinstance(replaced_by_filter_dict, dict)
                    filter_stats_dict = stats.get("filter_stats", {})
                    assert isinstance(filter_stats_dict, dict)
                    
                    for filter_name in applied_filters_list:
                        filter_info_dict = replaced_by_filter_dict.get(filter_name, {})
                        assert isinstance(filter_info_dict, dict)
                        filter_info = filter_info_dict
                        
                        # Универсальное обновление статистики
                        if filter_info["was_removed"]:
                            filter_stats_dict[f"{filter_name}_removed"] = filter_stats_dict.get(f"{filter_name}_removed", 0) + 1
                        else:
                            filter_stats_dict[f"{filter_name}_replaced"] = filter_stats_dict.get(f"{filter_name}_replaced", 0) + 1
                    
                    # Пропускаем строки, которые были удалены фильтрами
                    if not processed_line.strip():
                        continue
                    
                    # Проверяем дубликаты (если включена проверка) - сравниваем только с предыдущей строкой
                    if remove_duplicates and previous_line is not None:
                        if processed_line == previous_line:
                            dup_val = stats.get("duplicate_lines", 0)
                            assert isinstance(dup_val, int)
                            stats["duplicate_lines"] = dup_val + 1
                            continue  # Пропускаем дубликат предыдущей строки
                    
                    # Записываем обработанную строку
                    outfile.write(processed_line + '\n')
                    proc_val = stats.get("processed_lines", 0)
                    assert isinstance(proc_val, int)
                    stats["processed_lines"] = proc_val + 1
                    
                    # Обновляем предыдущую строку для следующей итерации
                    if remove_duplicates:
                        previous_line = processed_line
                    
                    # Обновляем прогресс
                    pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(line_num, total_lines)
    
    except Exception as e:
        raise Exception(f"Ошибка при обработке файла {input_path}: {e}")
    
    stats["processing_time"] = (datetime.now() - start_time).total_seconds()
    return stats


def create_output_filename(input_path: str, extension: str = ".lst") -> str:
    """
    Создает имя выходного файла на основе входного.
    
    Args:
        input_path: Путь к входному файлу
        extension: Расширение выходного файла
        
    Returns:
        Путь к выходному файлу
    """
    input_path_obj = Path(input_path)
    output_filename = input_path_obj.stem + extension
    return str(input_path_obj.parent / output_filename)


def ensure_directory_exists(path: str) -> None:
    """
    Создает директорию если она не существует.
    
    Args:
        path: Путь к директории
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_processing_stats(stats: Dict[str, Any], output_path: str) -> None:
    """
    Сохраняет статистику обработки в JSON файл.
    
    Args:
        stats: Словарь со статистикой
        output_path: Путь для сохранения статистики
    """
    stats["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def validate_file_exists(file_path: str) -> bool:
    """
    Проверяет существование файла.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        True если файл существует
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)


def get_file_size(file_path: str) -> int:
    """
    Получает размер файла в байтах.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Размер файла в байтах
    """
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """
    Форматирует размер файла в читаемый вид.
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        Отформатированная строка с размером
    """
    size: float = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def device_autoselect():
    """
    Автоматически выбирает устройство для обучения модели.
    
    Приоритет выбора: CUDA > MPS > CPU
    
    Returns:
        torch.device: Выбранное устройство (CUDA, MPS или CPU)
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def recommend_action(train_loss: float, val_loss: float, prev: dict | None) -> dict:
    """
    Принимает решение о продолжении или остановке обучения на основе метрик.
    
    Логика принятия решения:
    - Если это первая эпоха → продолжить
    - Если validation loss улучшился более чем на 0.01 → продолжить
    - Если val/train ratio > 1.2 → остановить (переобучение)
    - Иначе → остановить (нет дальнейшего улучшения)
    
    Args:
        train_loss: Средний training loss за эпоху
        val_loss: Средний validation loss за эпоху
        prev: Словарь с метриками предыдущей эпохи (или None для первой эпохи)
    
    Returns:
        dict: Рекомендация с ключами "action" ("continue" или "stop") и "reason"
    
    Example:
        >>> rec = recommend_action(2.5, 2.6, None)
        >>> rec["action"]
        'continue'
        >>> rec = recommend_action(2.0, 1.9, {"val_loss": 2.1})
        >>> rec["action"]
        'continue'
        >>> rec["reason"]
        'validation improved by 0.200'
    """
    # Первая эпоха - всегда продолжаем
    if not prev:
        return {"action": "continue", "reason": "first epoch"}
    
    # Вычисляем улучшение validation loss
    dval = prev["val_loss"] - val_loss
    
    # Вычисляем отношение val/train для обнаружения переобучения
    ratio = val_loss / max(train_loss, 1e-9)
    
    # Если validation loss значительно улучшился → продолжаем
    if dval > 0.01:
        return {"action": "continue", "reason": f"validation improved by {dval:.3f}"}
    
    # Если val_loss значительно больше train_loss → переобучение, останавливаем
    if ratio > 1.2:
        return {"action": "stop", "reason": f"overfitting (val/train={ratio:.2f})"}
    
    # Иначе → нет дальнейшего улучшения, останавливаем
    return {"action": "stop", "reason": "no further improvement"}


def ensure_dir(path: Path) -> None:
    """
    Создает директорию если она не существует (обертка для Path).
    
    Args:
        path: Путь к директории (Path объект)
    """
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> Iterable[dict]:
    """
    Загружает записи из JSONL файла построчно.
    
    Args:
        path: Путь к JSONL файлу
        
    Yields:
        Словари с данными из каждой строки
    """
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """
    Записывает записи в JSONL файл построчно.
    
    Args:
        path: Путь к выходному JSONL файлу
        records: Итерируемый объект со словарями для записи
    """
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
