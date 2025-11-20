#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95-analyze_drain.py
================
Анализирует работу алгоритма Drain3:
- Статистика по шаблонам и количеству строк на event_id
- Анализ качества кластеризации
- Топ самых частых шаблонов
- Сохранение детальной статистики
"""

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from drain3 import TemplateMiner # type: ignore[import-untyped]
from drain3.file_persistence import FilePersistence # type: ignore[import-untyped]

# Импортируем нашу систему конфигурации
from modules.config import get_config
from modules.logger import get_logger


def load_config_params() -> Dict[str, Any]:
    """
    Загружает параметры конфигурации для анализа Drain.
    
    Все файлы берутся из dataset.drain из config.yaml.
    
    Returns:
        Словарь с параметрами конфигурации
    """
    config = get_config()
    dataset_config = config.get("dataset", {})
    
    # Базовая директория для всех drain-файлов из dataset.drain
    drain_dir = Path(dataset_config.get("drain", "./dataset/drain"))
    drain_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "drain_dir": drain_dir,
        "prepared_dir": Path(dataset_config.get("prepared", "./dataset/prepared")),
        # Все drain-файлы находятся в drain_dir
        "state_file": drain_dir / "drain_state.json",
        "sequences_file": drain_dir / "drain_sequences.jsonl",
        "templates_file": drain_dir / "drain_templates.json",
        "stats_file": drain_dir / "drain_analysis_stats.json"
    }


def load_drain_state(config: Dict[str, Any]) -> Tuple[TemplateMiner, Dict[str, str]]:
    """
    Загружает состояние Drain3 и извлекает шаблоны.
    
    Args:
        config: Конфигурация с путями к файлам
        
    Returns:
        Кортеж (template_miner, templates_dict)
    """
    logger = get_logger("analyze_drain", "analyze_drain")
    
    # Проверяем существование файла состояния
    if not config["state_file"].exists():
        raise FileNotFoundError(f"Файл состояния Drain3 не найден: {config['state_file']}")
    
    logger.info(f"Загружаем состояние Drain3 из {config['state_file']}")
    
    # Загружаем состояние Drain3
    persistence = FilePersistence(str(config["state_file"]))
    template_miner = TemplateMiner(persistence)
    
    # Извлекаем шаблоны
    clusters = template_miner.drain.clusters
    templates = {}
    
    for cluster in clusters:
        template = cluster.get_template()
        templates[f"E{cluster.cluster_id}"] = template
    
    logger.info(f"Загружено {len(templates)} шаблонов")
    return template_miner, templates


def analyze_event_frequencies(config: Dict[str, Any], templates: Dict[str, str]) -> Dict[str, Any]:
    """
    Анализирует частоты событий из файла последовательностей.
    
    Args:
        config: Конфигурация с путями к файлам
        templates: Словарь шаблонов {event_id: template}
        
    Returns:
        Словарь со статистикой частот
    """
    logger = get_logger("analyze_drain", "analyze_drain")
    
    # Проверяем существование файла последовательностей
    if not config["sequences_file"].exists():
        raise FileNotFoundError(f"Файл последовательностей не найден: {config['sequences_file']}")
    
    logger.info(f"Анализируем частоты событий из {config['sequences_file']}")
    
    # Считаем частоты event_id
    event_counter: Counter[str] = Counter()
    pipeline_stats: Dict[str, int] = {}
    build_stats: Dict[str, int] = {}
    
    with open(config["sequences_file"], "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        with tqdm(total=len(lines), desc="Анализ последовательностей") as pbar:
            for line in lines:
                rec = json.loads(line.strip())
                pipeline_id = rec.get("pipeline_id", "unknown")
                build_id = rec.get("build_id", "unknown")
                event_seq = rec.get("event_seq", [])
                
                # Обновляем статистику по pipeline и build
                pipeline_stats[pipeline_id] = pipeline_stats.get(pipeline_id, 0) + len(event_seq)
                build_stats[build_id] = build_stats.get(build_id, 0) + len(event_seq)
                
                # Считаем частоты событий
                for eid in event_seq:
                    event_counter[eid] += 1
                
                pbar.update(1)
    
    total_events = sum(event_counter.values())
    unique_events = len(event_counter)
    
    return {
        "total_events": total_events,
        "unique_events": unique_events,
        "avg_events_per_template": total_events / unique_events if unique_events > 0 else 0,
        "event_counter": event_counter,
        "pipeline_stats": pipeline_stats,
        "build_stats": build_stats,
        "total_sequences": len(lines)
    }


def analyze_template_quality(templates: Dict[str, str], event_counter: Counter) -> Dict[str, Any]:
    """
    Анализирует качество извлеченных шаблонов.
    
    Args:
        templates: Словарь шаблонов
        event_counter: Счетчик частот событий
        
    Returns:
        Словарь с анализом качества
    """
    logger = get_logger("analyze_drain", "analyze_drain")
    logger.info("Анализируем качество шаблонов")
    
    # Статистика по длине шаблонов
    template_lengths = [len(template.split()) for template in templates.values()]
    
    # Статистика по частоте шаблонов
    template_frequencies = list(event_counter.values())
    
    # Анализ параметризации (количество <*> в шаблонах)
    parameterized_templates = 0
    parameter_counts = []
    
    for template in templates.values():
        param_count = template.count("<*>")
        parameter_counts.append(param_count)
        if param_count > 0:
            parameterized_templates += 1
    
    # Статистика по уникальности шаблонов
    unique_templates = len(set(templates.values()))
    duplicate_templates = len(templates) - unique_templates
    
    return {
        "template_length_stats": {
            "min": min(template_lengths) if template_lengths else 0,
            "max": max(template_lengths) if template_lengths else 0,
            "avg": sum(template_lengths) / len(template_lengths) if template_lengths else 0
        },
        "frequency_stats": {
            "min": min(template_frequencies) if template_frequencies else 0,
            "max": max(template_frequencies) if template_frequencies else 0,
            "avg": sum(template_frequencies) / len(template_frequencies) if template_frequencies else 0
        },
        "parameterization_stats": {
            "parameterized_count": parameterized_templates,
            "parameterized_ratio": parameterized_templates / len(templates) if templates else 0,
            "avg_parameters": sum(parameter_counts) / len(parameter_counts) if parameter_counts else 0
        },
        "uniqueness_stats": {
            "unique_templates": unique_templates,
            "duplicate_templates": duplicate_templates,
            "uniqueness_ratio": unique_templates / len(templates) if templates else 0
        }
    }


def print_analysis_report(templates: Dict[str, str], freq_stats: Dict[str, Any], 
                         quality_stats: Dict[str, Any], top_n: int = 20) -> None:
    """
    Выводит отчет анализа в консоль.
    
    Args:
        templates: Словарь шаблонов
        freq_stats: Статистика частот
        quality_stats: Статистика качества
        top_n: Количество топ шаблонов для показа
    """
    print("=" * 80)
    print("📊 АНАЛИЗ РАБОТЫ АЛГОРИТМА DRAIN3")
    print("=" * 80)
    
    # Общая статистика
    print(f"✅ Загружено шаблонов: {len(templates)}")
    print(f"📊 Всего событий: {freq_stats['total_events']:,}")
    print(f"📦 Уникальных event_id: {freq_stats['unique_events']:,}")
    print(f"📉 Среднее количество строк на шаблон: {freq_stats['avg_events_per_template']:.2f}")
    print(f"🔄 Всего последовательностей: {freq_stats['total_sequences']:,}")
    
    print("\n" + "-" * 80)
    print("📈 СТАТИСТИКА КАЧЕСТВА ШАБЛОНОВ")
    print("-" * 80)
    
    # Статистика длины шаблонов
    len_stats = quality_stats["template_length_stats"]
    print(f"📏 Длина шаблонов: мин={len_stats['min']}, макс={len_stats['max']}, среднее={len_stats['avg']:.1f}")
    
    # Статистика частоты
    freq_stats_detail = quality_stats["frequency_stats"]
    print(f"🔢 Частота шаблонов: мин={freq_stats_detail['min']}, макс={freq_stats_detail['max']}, среднее={freq_stats_detail['avg']:.1f}")
    
    # Статистика параметризации
    param_stats = quality_stats["parameterization_stats"]
    print(f"🔧 Параметризованных шаблонов: {param_stats['parameterized_count']} ({param_stats['parameterized_ratio']:.1%})")
    print(f"📊 Среднее количество параметров: {param_stats['avg_parameters']:.1f}")
    
    # Статистика уникальности
    unique_stats = quality_stats["uniqueness_stats"]
    print(f"🎯 Уникальных шаблонов: {unique_stats['unique_templates']} ({unique_stats['uniqueness_ratio']:.1%})")
    print(f"🔄 Дублированных шаблонов: {unique_stats['duplicate_templates']}")
    
    print("\n" + "-" * 80)
    print(f"🔝 ТОП-{top_n} САМЫХ ЧАСТЫХ ШАБЛОНОВ")
    print("-" * 80)
    
    # Топ шаблонов
    event_counter = freq_stats["event_counter"]
    for i, (eid, count) in enumerate(event_counter.most_common(top_n), 1):
        template = templates.get(eid, "<unknown>")
        percentage = (count / freq_stats["total_events"]) * 100
        print(f"{i:2d}. {eid:>6}  ×{count:<6}  ({percentage:5.2f}%)  {template}")
    
    print("\n" + "-" * 80)
    print("📋 СТАТИСТИКА ПО PIPELINE")
    print("-" * 80)
    
    # Топ pipeline по количеству событий
    pipeline_stats = freq_stats["pipeline_stats"]
    top_pipelines = sorted(pipeline_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for i, (pipeline_id, count) in enumerate(top_pipelines, 1):
        percentage = (count / freq_stats["total_events"]) * 100
        print(f"{i:2d}. Pipeline {pipeline_id}: {count:,} событий ({percentage:.2f}%)")


def save_detailed_stats(config: Dict[str, Any], templates: Dict[str, str], 
                       freq_stats: Dict[str, Any], quality_stats: Dict[str, Any], 
                       top_n: int = 50) -> None:
    """
    Сохраняет детальную статистику в JSON файл.
    
    Args:
        config: Конфигурация с путями к файлам
        templates: Словарь шаблонов
        freq_stats: Статистика частот
        quality_stats: Статистика качества
        top_n: Количество топ шаблонов для сохранения
    """
    logger = get_logger("analyze_drain", "analyze_drain")
    
    # Подготавливаем данные для сохранения
    event_counter = freq_stats["event_counter"]
    
    detailed_stats = {
        "analysis_metadata": {
            "total_templates": len(templates),
            "total_events": freq_stats["total_events"],
            "unique_events": freq_stats["unique_events"],
            "total_sequences": freq_stats["total_sequences"],
            "analysis_timestamp": str(Path().cwd())
        },
        "frequency_analysis": {
            "avg_events_per_template": freq_stats["avg_events_per_template"],
            "pipeline_stats": freq_stats["pipeline_stats"],
            "build_stats": freq_stats["build_stats"]
        },
        "quality_analysis": quality_stats,
        "top_templates": [
            {
                "rank": i,
                "event_id": eid,
                "count": count,
                "percentage": (count / freq_stats["total_events"]) * 100,
                "template": templates.get(eid, "<unknown>")
            }
            for i, (eid, count) in enumerate(event_counter.most_common(top_n), 1)
        ],
        "template_distribution": {
            "frequency_ranges": {
                "1": sum(1 for count in event_counter.values() if count == 1),
                "2-5": sum(1 for count in event_counter.values() if 2 <= count <= 5),
                "6-10": sum(1 for count in event_counter.values() if 6 <= count <= 10),
                "11-50": sum(1 for count in event_counter.values() if 11 <= count <= 50),
                "51-100": sum(1 for count in event_counter.values() if 51 <= count <= 100),
                "100+": sum(1 for count in event_counter.values() if count > 100)
            }
        }
    }
    
    # Сохраняем в файл
    with open(config["stats_file"], "w", encoding="utf-8") as f:
        json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Детальная статистика сохранена → {config['stats_file']}")
    print(f"\n💾 Детальная статистика сохранена → {config['stats_file']}")


def parse_arguments() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    
    Returns:
        Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="Анализ работы алгоритма Drain3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python3 analyze_drain.py
  python3 analyze_drain.py --top 30
  python3 analyze_drain.py --top 50 --verbose
        """
    )
    
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=20,
        help="Количество топ шаблонов для показа (по умолчанию: 20)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    return parser.parse_args()


def main():
    """Основная функция скрипта анализа Drain."""
    args = parse_arguments()
    
    # Инициализируем логгер
    logger = get_logger("analyze_drain", "analyze_drain")
    
    try:
        logger.info("=== Запуск анализа работы алгоритма Drain3 ===")
        
        # Загружаем конфигурацию
        config = load_config_params()
        logger.info(f"Конфигурация загружена: drain_dir={config['drain_dir']}")
        
        # Загружаем состояние Drain3
        template_miner, templates = load_drain_state(config)
        
        # Анализируем частоты событий
        freq_stats = analyze_event_frequencies(config, templates)
        
        # Анализируем качество шаблонов
        quality_stats = analyze_template_quality(templates, freq_stats["event_counter"])
        
        # Выводим отчет
        print_analysis_report(templates, freq_stats, quality_stats, args.top)
        
        # Сохраняем детальную статистику
        save_detailed_stats(config, templates, freq_stats, quality_stats, args.top * 2)
        
        logger.info("=== Анализ завершен успешно ===")
        print(f"\n✅ Анализ работы Drain3 завершен успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при анализе Drain3: {e}")
        print(f"\n❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
