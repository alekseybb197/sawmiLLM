#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
94-check_drain_sequence.py
Проверка целостности Drain-последовательностей событий.

Скрипт проверяет целостность последовательностей событий в файле drain_sequences.jsonl,
проверяя первую и последнюю последовательности для выявления нарушений целостности.

Проверяет:
  - Все event_id из последовательностей существуют в drain_templates.json
  - Последовательности логичны и не содержат ошибок

Примеры запуска:
  # Проверка целостности первой и последней последовательностей
  ./94-check_drain_sequence.py

Требуемые файлы (из config.yaml):
  - dataset.drain/drain_templates.json - словарь шаблонов событий
  - dataset.vocab_list (первый файл) - drain_sequences.jsonl
"""

import typer
from pathlib import Path
from typing import List, Dict, Any

from modules.config import get_config
from modules.decode import load_jsonl, load_json, decode_sequence


def check_sequence_integrity(
    rec: Dict[Any, Any],
    templates: Dict[str, str],
) -> Dict[str, Any]:
    """
    Проверяет целостность одной последовательности событий.
    
    Args:
        rec: Запись из drain_sequences.jsonl
        templates: Словарь шаблонов {event_id: template_string}
        
    Returns:
        Словарь с результатами проверки
    """
    event_seq = rec.get("event_seq", [])
    pipeline_id = rec.get("pipeline_id", "—")
    build_id = rec.get("build_id", "—")
    label = rec.get("label", "—")
    
    # Проверяем наличие всех event_id в templates
    unknown_events = []
    for eid in event_seq:
        if eid not in templates:
            unknown_events.append(eid)
    
    # Расшифровываем последовательность
    decoded = decode_sequence(event_seq, templates)
    
    return {
        "pipeline_id": pipeline_id,
        "build_id": build_id,
        "label": label,
        "event_seq": event_seq,
        "decoded": decoded,
        "unknown_events": unknown_events,
        "is_valid": len(unknown_events) == 0,
    }


# Создаем приложение Typer
app = typer.Typer(
    name="check_drain_sequence",
    help="Проверка целостности Drain-последовательностей",
    add_completion=False,
)


@app.command()
def main() -> None:
    """
    Проверяет целостность первой и последней последовательностей в drain_sequences.jsonl.
    
    Проверяет:
      - Все event_id существуют в drain_templates.json
      - Последовательности логичны и не содержат ошибок
    """
    # Загружаем конфигурацию
    config = get_config()
    dataset_cfg = config.get("dataset", {})
    
    # Получаем путь к drain_sequences.jsonl из vocab_list
    vocab_list = dataset_cfg.get("vocab_list", [])
    if not vocab_list:
        print("❌ Ошибка: vocab_list не найден в config.yaml")
        print("   Проверьте секцию dataset.vocab_list в config.yaml")
        raise typer.Exit(code=1)
    
    sequences_path = Path(vocab_list[0])
    if not sequences_path.exists():
        print(f"❌ Файл последовательностей не найден: {sequences_path}")
        raise typer.Exit(code=1)
    
    # Получаем путь к drain_templates.json
    base_dir = dataset_cfg.get("drain", "./dataset/drain")
    base_dir_path = Path(base_dir)
    templates_path = base_dir_path / "drain_templates.json"
    
    # Загружаем словарь шаблонов
    print(f"📥 Загрузка шаблонов: {templates_path}")
    if not templates_path.exists():
        print(f"❌ Файл шаблонов не найден: {templates_path}")
        raise typer.Exit(code=1)
    templates = load_json(templates_path)
    print(f"✅ Загружено {len(templates)} шаблонов.")
    
    # Загружаем последовательности
    print(f"📥 Загрузка последовательностей: {sequences_path}")
    sequences = load_jsonl(sequences_path)
    if not sequences:
        print(f"❌ Файл {sequences_path} пуст.")
        raise typer.Exit(code=1)
    print(f"✅ Загружено {len(sequences)} последовательностей.")
    
    # Проверяем первую и последнюю последовательности
    print(f"\n{'='*90}")
    print(f"🔍 Проверка целостности последовательностей")
    print(f"{'='*90}")
    
    # Первая последовательность
    first_rec = sequences[0]
    print(f"\n📋 Первая последовательность (индекс 0):")
    first_result = check_sequence_integrity(first_rec, templates)
    
    print(f"   pipeline_id: {first_result['pipeline_id']}")
    print(f"   build_id: {first_result['build_id']}")
    print(f"   label: {first_result['label']}")
    print(f"   Количество событий: {len(first_result['event_seq'])}")
    
    if first_result['unknown_events']:
        print(f"   ❌ Найдены неизвестные события: {first_result['unknown_events']}")
    else:
        print(f"   ✅ Все события найдены в словаре шаблонов")
    
    print(f"   event_seq: {first_result['event_seq'][:10]}..." if len(first_result['event_seq']) > 10 else f"   event_seq: {first_result['event_seq']}")
    print(f"\n   Расшифровка шаблонов (первые 5):")
    for i, template in enumerate(first_result['decoded'][:5], 1):
        print(f"     {i:02d}. {template}")
    
    # Последняя последовательность
    if len(sequences) > 1:
        last_rec = sequences[-1]
        print(f"\n📋 Последняя последовательность (индекс {len(sequences) - 1}):")
        last_result = check_sequence_integrity(last_rec, templates)
        
        print(f"   pipeline_id: {last_result['pipeline_id']}")
        print(f"   build_id: {last_result['build_id']}")
        print(f"   label: {last_result['label']}")
        print(f"   Количество событий: {len(last_result['event_seq'])}")
        
        if last_result['unknown_events']:
            print(f"   ❌ Найдены неизвестные события: {last_result['unknown_events']}")
        else:
            print(f"   ✅ Все события найдены в словаре шаблонов")
        
        print(f"   event_seq: {last_result['event_seq'][:10]}..." if len(last_result['event_seq']) > 10 else f"   event_seq: {last_result['event_seq']}")
        print(f"\n   Расшифровка шаблонов (первые 5):")
        for i, template in enumerate(last_result['decoded'][:5], 1):
            print(f"     {i:02d}. {template}")
    else:
        print(f"\n⚠️  В файле только одна последовательность, проверяется только она.")
    
    # Итоговая проверка целостности
    print(f"\n{'='*90}")
    print(f"📊 Итоговая проверка целостности")
    print(f"{'='*90}")
    
    all_unknown = []
    for rec in sequences:
        event_seq = rec.get("event_seq", [])
        for eid in event_seq:
            if eid not in templates:
                if eid not in all_unknown:
                    all_unknown.append(eid)
    
    if all_unknown:
        print(f"❌ Найдены неизвестные события в последовательностях:")
        for eid in all_unknown:
            print(f"   - {eid}")
    else:
        print(f"✅ Все события из последовательностей найдены в словаре шаблонов")
    
    # Проверяем уникальность event_id в последовательностях
    total_sequences = len(sequences)
    unique_pipelines = set()
    for rec in sequences:
        pipeline_id = rec.get("pipeline_id")
        if pipeline_id:
            unique_pipelines.add(str(pipeline_id))
    
    print(f"\n📈 Статистика:")
    print(f"   • Всего последовательностей: {total_sequences}")
    print(f"   • Уникальных пайплайнов: {len(unique_pipelines)}")
    print(f"   • Всего шаблонов в словаре: {len(templates)}")
    
    if all_unknown:
        raise typer.Exit(code=1)
    else:
        print(f"\n✅ Проверка целостности завершена успешно")


if __name__ == "__main__":
    app()
