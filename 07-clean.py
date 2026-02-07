#!/usr/bin/env python3
"""
Скрипт для первого этапа очистки логов.
Удаляет очевидно незначимые строки из файлов логов.

Использование:
    python clean.py <input_file>
    
Пример:
    python clean.py 204923-26492182-logs_content.txt
"""

import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

from modules.config import get_config
from modules.logger import get_logger
from modules.utils import (
    process_log_file, 
    create_output_filename, 
    save_processing_stats,
    validate_file_exists,
    get_file_size,
    format_file_size,
    ensure_directory_exists
)


def parse_arguments() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    
    Returns:
        Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для первого этапа очистки логов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python clean.py 204923-26492182-logs_content.txt
  python clean.py /path/to/logs.txt --output /path/to/output.raw
        """
    )
    
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Путь к входному файлу логов (.txt) или не указывать для пакетной обработки"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Путь к выходному файлу (по умолчанию: <input>.raw)"
    )
    
    parser.add_argument(
        "--stats", "-s",
        help="Путь для сохранения статистики обработки (JSON)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    return parser.parse_args()


def validate_input_file(input_path: str) -> None:
    """
    Проверяет входной файл.
    
    Args:
        input_path: Путь к входному файлу
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если файл пустой или недоступен
    """
    if not validate_file_exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    
    file_size = get_file_size(input_path)
    if file_size == 0:
        raise ValueError(f"Файл пустой: {input_path}")
    
    print(f"Входной файл: {input_path}")
    print(f"Размер файла: {format_file_size(file_size)}")


def get_batch_files(config: Dict[str, Any]) -> List[str]:
    """
    Получает список файлов для пакетной обработки из dataset.original.
    
    Args:
        config: Конфигурация датасета
        
    Returns:
        Список путей к файлам .txt
    """
    original_dir = config.get("original", "./dataset/original")
    
    # Проверяем существование директории
    if not Path(original_dir).exists():
        raise FileNotFoundError(f"Директория не найдена: {original_dir}")
    
    # Ищем все файлы .txt в директории
    pattern = str(Path(original_dir) / "*.txt")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"Файлы .txt не найдены в директории: {original_dir}")
    
    return sorted(files)


def process_batch_files(config: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    Обрабатывает все файлы в пакетном режиме.
    
    Args:
        config: Конфигурация датасета
        logger: Логгер
        
    Returns:
        Общая статистика обработки
    """
    # Получаем список файлов
    input_files = get_batch_files(config)
    cleaned_dir = config.get("cleaned", "./dataset/cleaned")
    
    # Создаем выходную директорию
    ensure_directory_exists(str(Path(cleaned_dir)))
    
    logger.info(f"Найдено файлов для обработки: {len(input_files)}")
    logger.info(f"Входная директория: {config.get('original', './dataset/original')}")
    logger.info(f"Выходная директория: {cleaned_dir}")
    
    # Общая статистика
    batch_stats: Dict[str, Any] = {
        "total_files": len(input_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_lines": 0,
        "total_processed_lines": 0,
        "total_duplicate_lines": 0,
        "total_processing_time": 0.0,
        "file_stats": []
    }
    
    # Обрабатываем каждый файл
    for i, input_file in enumerate(input_files, 1):
        try:
            logger.info(f"=== Обработка файла {i}/{len(input_files)}: {Path(input_file).name} ===")
            
            # Определяем выходной файл
            input_filename = Path(input_file).stem
            output_file = Path(cleaned_dir) / f"{input_filename}.raw"
            
            # Обрабатываем файл
            stats = process_log_file(input_file, str(output_file), pattern_type="clean_patterns", remove_duplicates=True)
            
            # Обновляем общую статистику
            processed_files_val = batch_stats.get("processed_files", 0)
            assert isinstance(processed_files_val, int)
            batch_stats["processed_files"] = processed_files_val + 1
            
            total_lines_val = batch_stats.get("total_lines", 0)
            assert isinstance(total_lines_val, int)
            batch_stats["total_lines"] = total_lines_val + stats.get("total_lines", 0)
            
            total_processed_lines_val = batch_stats.get("total_processed_lines", 0)
            assert isinstance(total_processed_lines_val, int)
            batch_stats["total_processed_lines"] = total_processed_lines_val + stats.get("processed_lines", 0)
            
            total_duplicate_lines_val = batch_stats.get("total_duplicate_lines", 0)
            assert isinstance(total_duplicate_lines_val, int)
            batch_stats["total_duplicate_lines"] = total_duplicate_lines_val + stats.get("duplicate_lines", 0)
            
            total_time_val = batch_stats.get("total_processing_time", 0.0)
            assert isinstance(total_time_val, (int, float))
            batch_stats["total_processing_time"] = total_time_val + stats.get("processing_time", 0.0)
            
            # Сохраняем статистику файла
            file_stat: Dict[str, Any] = {
                "input_file": input_file,
                "output_file": str(output_file),
                "total_lines": stats.get("total_lines", 0),
                "processed_lines": stats.get("processed_lines", 0),
                "duplicate_lines": stats.get("duplicate_lines", 0),
                "processing_time": stats.get("processing_time", 0.0)
            }
            file_stats_list = batch_stats.get("file_stats", [])
            assert isinstance(file_stats_list, list)
            file_stats_list.append(file_stat)
            
            logger.info(f"✅ Файл обработан: {stats['processed_lines']}/{stats['total_lines']} строк, "
                       f"дубликатов: {stats['duplicate_lines']}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке файла {input_file}: {e}")
            failed_files_val = batch_stats.get("failed_files", 0)
            assert isinstance(failed_files_val, int)
            batch_stats["failed_files"] = failed_files_val + 1
    
    return batch_stats


def setup_output_path(input_path: str, output_arg: Optional[str] = None) -> str:
    """
    Настраивает путь к выходному файлу.
    
    Args:
        input_path: Путь к входному файлу
        output_arg: Аргумент --output из командной строки
        
    Returns:
        Путь к выходному файлу
    """
    if output_arg:
        output_path = output_arg
    else:
        output_path = create_output_filename(input_path, ".raw")
    
    # Создаем директорию если нужно
    ensure_directory_exists(str(Path(output_path).parent))
    
    return output_path


def check_data_quality(input_path: str, output_path: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Проверяет качество очищенных данных.
    
    Args:
        input_path: Путь к входному файлу
        output_path: Путь к выходному файлу
        stats: Статистика обработки
        
    Returns:
        Словарь с результатами проверки качества
    """
    quality_results: Dict[str, Any] = {
        "input_file_size": get_file_size(input_path),
        "output_file_size": get_file_size(output_path),
        "compression_ratio": 0.0,
        "processing_efficiency": 0.0,
        "data_integrity": True,
        "issues": []
    }
    
    # Вычисляем коэффициент сжатия
    input_size = quality_results.get("input_file_size", 0)
    assert isinstance(input_size, int)
    if input_size > 0:
        output_size = quality_results.get("output_file_size", 0)
        assert isinstance(output_size, int)
        quality_results["compression_ratio"] = float(output_size) / float(input_size)
    
    # Вычисляем эффективность обработки
    total_lines = stats.get("total_lines", 0)
    assert isinstance(total_lines, int)
    if total_lines > 0:
        processed_lines = stats.get("processed_lines", 0)
        assert isinstance(processed_lines, int)
        quality_results["processing_efficiency"] = float(processed_lines) / float(total_lines)
    
    # Проверяем целостность данных
    processed_lines_check = stats.get("processed_lines", 0)
    assert isinstance(processed_lines_check, int)
    if processed_lines_check == 0:
        quality_results["data_integrity"] = False
        issues_list = quality_results.get("issues", [])
        assert isinstance(issues_list, list)
        issues_list.append("Не обработано ни одной строки")
    
    # Проверяем эффективность обработки
    efficiency = quality_results.get("processing_efficiency", 0.0)
    assert isinstance(efficiency, (int, float))
    if float(efficiency) < 0.5:
        issues_list = quality_results.get("issues", [])
        assert isinstance(issues_list, list)
        issues_list.append("Низкая эффективность очистки данных")
    
    return quality_results


def main():
    """Основная функция скрипта."""
    # Парсим аргументы командной строки
    args = parse_arguments()
    
    # Инициализируем конфигурацию и логгер
    config_obj = get_config()
    dataset_config = config_obj.get("dataset", {})
    logger = get_logger("clean", "clean")
    
    try:
        logger.info("=== Запуск скрипта очистки логов ===")
        
        # Определяем режим работы
        if args.input_file:
            # Режим обработки одного файла
            logger.info(f"Режим: обработка одного файла")
            logger.info(f"Входной файл: {args.input_file}")
            
            # Проверяем входной файл
            validate_input_file(args.input_file)
            
            # Настраиваем выходной файл
            output_path = setup_output_path(args.input_file, args.output)
            logger.info(f"Выходной файл: {output_path}")
            
            # Обрабатываем файл
            logger.info("Начинаем очистку файла...")
            stats = process_log_file(args.input_file, output_path, pattern_type="clean_patterns", remove_duplicates=True)
            
            # Логируем статистику
            logger.info("=== Статистика очистки ===")
            logger.info(f"Всего строк: {stats['total_lines']}")
            logger.info(f"Обработано строк: {stats['processed_lines']}")
            logger.info(f"Удалено дубликатов: {stats['duplicate_lines']}")
            logger.info(f"Время обработки: {stats['processing_time']:.2f} сек")
            
            # Логируем детальную статистику фильтров
            if 'filter_stats' in stats and stats['filter_stats']:
                logger.info("=== Статистика фильтров очистки ===")
                for filter_name, count in stats['filter_stats'].items():
                    logger.info(f"{filter_name}: {count}")
            else:
                logger.info("=== Статистика фильтров очистки ===")
                logger.info("Фильтры очистки не применялись")
            
            # Проверяем качество данных
            logger.info("Проверяем качество очищенных данных...")
            quality_results = check_data_quality(args.input_file, output_path, stats)
            
            logger.info("=== Результаты проверки качества ===")
            logger.info(f"Коэффициент сжатия: {quality_results['compression_ratio']:.2f}")
            logger.info(f"Эффективность очистки: {quality_results['processing_efficiency']:.2f}")
            logger.info(f"Целостность данных: {quality_results['data_integrity']}")
            
            if quality_results["issues"]:
                logger.warning("Обнаружены проблемы:")
                for issue in quality_results["issues"]:
                    logger.warning(f"  - {issue}")
            
            # Сохраняем статистику
            if args.stats:
                stats_path = args.stats
            else:
                stats_path = create_output_filename(args.input_file, "_clean_stats.json")
            
            # Объединяем статистику и результаты качества
            full_stats = {**stats, "quality_check": quality_results}
            save_processing_stats(full_stats, stats_path)
            logger.info(f"Статистика сохранена: {stats_path}")
            
            # Логируем метрики для JSON лога
            from modules.logger import Logger
            logger_instance = Logger("clean", "clean")
            logger_instance.log_metrics(stats)
            logger_instance.log_data_quality(quality_results)
            
            logger.info("=== Очистка завершена успешно ===")
            print(f"\n✅ Файл успешно очищен!")
            print(f"📁 Выходной файл: {output_path}")
            print(f"📊 Статистика: {stats_path}")
            
        else:
            # Режим пакетной обработки
            logger.info("Режим: пакетная обработка")
            
            # Обрабатываем все файлы
            batch_stats = process_batch_files(dataset_config, logger)
            
            # Логируем общую статистику
            logger.info("=== Общая статистика пакетной обработки ===")
            logger.info(f"Всего файлов: {batch_stats['total_files']}")
            logger.info(f"Обработано файлов: {batch_stats['processed_files']}")
            logger.info(f"Ошибок: {batch_stats['failed_files']}")
            logger.info(f"Всего строк: {batch_stats['total_lines']}")
            logger.info(f"Обработано строк: {batch_stats['total_processed_lines']}")
            logger.info(f"Удалено дубликатов: {batch_stats['total_duplicate_lines']}")
            logger.info(f"Общее время обработки: {batch_stats['total_processing_time']:.2f} сек")
            
            # Сохраняем общую статистику
            stats_path = "batch_clean_stats.json"
            save_processing_stats(batch_stats, stats_path)
            logger.info(f"Общая статистика сохранена: {stats_path}")
            
            logger.info("=== Пакетная очистка завершена успешно ===")
            print(f"\n✅ Пакетная обработка завершена!")
            print(f"📁 Обработано файлов: {batch_stats['processed_files']}/{batch_stats['total_files']}")
            print(f"📊 Общая статистика: {stats_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {e}")
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
