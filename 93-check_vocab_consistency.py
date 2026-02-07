#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
93-check_vocab_consistency.py
Проверка согласованности Drain-индекса и словаря событий модели.

Использует настройки из config.yaml:
  - dataset.drain - директория с drain файлами
  - dataset.vocab - директория со словарем событий
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, cast

from modules.config import get_config


def load_json(path: Path) -> Dict[Any, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast(Dict[Any, Any], json.load(f))


def load_jsonl(path: Path) -> List[Dict[Any, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [cast(Dict[Any, Any], json.loads(line)) for line in f if line.strip()]


def collect_event_ids_from_sequences(seq_path: Path) -> Set[str]:
    seqs = load_jsonl(seq_path)
    all_events: Set[str] = set()
    for rec in seqs:
        seq = rec.get("event_seq") or []
        all_events.update(seq)
    return all_events


def main():
    parser = argparse.ArgumentParser(
        description="Проверка согласованности Drain-индекса и словаря событий модели"
    )
    parser.add_argument(
        "--drain-dir",
        type=str,
        help="Директория с drain файлами (переопределяет config.yaml dataset.drain)"
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        help="Директория со словарем (переопределяет config.yaml dataset.vocab)"
    )
    parser.add_argument(
        "--drain-templates",
        type=str,
        help="Путь к drain_templates.json (переопределяет автоматический путь)"
    )
    parser.add_argument(
        "--event-vocab",
        type=str,
        help="Путь к event_vocab.json (переопределяет автоматический путь)"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        help="Путь к drain_sequences.jsonl (переопределяет автоматический путь)"
    )
    args = parser.parse_args()

    # --- Загрузка конфигурации ---
    config = get_config()
    dataset_cfg = config.get_dataset_config()
    
    # Определяем пути из конфига или аргументов
    if args.drain_dir:
        drain_dir = Path(args.drain_dir)
    else:
        drain_dir = Path(dataset_cfg.get("drain", "./dataset/drain"))
    
    if args.vocab_dir:
        vocab_dir = Path(args.vocab_dir)
    else:
        vocab_dir = Path(dataset_cfg.get("vocab", "./dataset/vocab"))
    
    # Пути к файлам
    if args.drain_templates:
        drain_templates_path = Path(args.drain_templates)
    else:
        drain_templates_path = drain_dir / "drain_templates.json"
    
    if args.event_vocab:
        event_vocab_path = Path(args.event_vocab)
    else:
        event_vocab_path = vocab_dir / "event_vocab.json"
    
    if args.sequences:
        sequences_path = Path(args.sequences)
    else:
        sequences_path = drain_dir / "drain_sequences.jsonl"

    print(f"📁 Drain директория: {drain_dir}")
    print(f"📁 Vocab директория: {vocab_dir}")
    print(f"📄 drain_templates.json: {drain_templates_path}")
    print(f"📄 event_vocab.json: {event_vocab_path}")
    print(f"📄 drain_sequences.jsonl: {sequences_path}")

    # --- Проверка наличия файлов ---
    missing_files = []
    for name, path in [
        ("drain_templates.json", drain_templates_path),
        ("event_vocab.json", event_vocab_path),
        ("drain_sequences.jsonl", sequences_path),
    ]:
        if not path.exists():
            missing_files.append((name, path))
    
    if missing_files:
        print("\n❌ Не найдены файлы:")
        for name, path in missing_files:
            print(f"   {name}: {path}")
        return

    # --- Загрузка данных ---
    print("📥 Загрузка файлов...")
    drain = load_json(drain_templates_path)
    vocab = load_json(event_vocab_path)
    used = collect_event_ids_from_sequences(sequences_path)

    drain_ids = set(drain.keys())
    vocab_ids = set(vocab.keys())

    # --- Анализ ---
    unused_in_sequences = drain_ids - used
    missing_in_vocab = used - vocab_ids
    orphan_in_vocab = vocab_ids - drain_ids

    print("\n🧩 === Статистика ===")
    print(f"🔹 Всего шаблонов в Drain: {len(drain_ids)}")
    print(f"🔹 Всего событий в словаре: {len(vocab_ids)}")
    print(f"🔹 Всего событий в последовательностях: {len(used)}")

    print("\n📊 === Несоответствия ===")
    print(f"⚠️  Событий есть в Drain, но они ни разу не встречаются в drain_sequences: {len(unused_in_sequences)}")
    print(f"⚠️  Событий встречаются в drain_sequences, но отсутствуют в event_vocab: {len(missing_in_vocab)}")
    print(f"⚠️  Событий есть в event_vocab, но нет в drain_templates: {len(orphan_in_vocab)}")

    # --- Примеры с расшифровкой шаблонов ---
    def show_examples(title: str, items: Set[str], drain_dict: Dict[str, str], vocab_dict: Dict[str, Any]):
        if not items:
            print(f"✅ {title}: всё в порядке.")
            return
        examples = list(sorted(items))[:10]
        print(f"\n🔍 {title} (первые 10):")
        for eid in examples:
            # Пытаемся найти шаблон в drain
            template = drain_dict.get(eid)
            if template:
                # Обрезаем длинные шаблоны для читаемости
                template_preview = template[:80] + "..." if len(template) > 80 else template
                print(f"  {eid:>8} → {template_preview}")
            else:
                # Если нет в drain, пробуем vocab
                vocab_entry = vocab_dict.get(eid)
                if vocab_entry:
                    if isinstance(vocab_entry, dict):
                        vocab_preview = str(vocab_entry)[:80] + "..." if len(str(vocab_entry)) > 80 else str(vocab_entry)
                    else:
                        vocab_preview = str(vocab_entry)[:80] + "..." if len(str(vocab_entry)) > 80 else str(vocab_entry)
                    print(f"  {eid:>8} → [только в vocab] {vocab_preview}")
                else:
                    print(f"  {eid:>8} → [шаблон не найден]")

    show_examples("Неиспользуемые шаблоны Drain", unused_in_sequences, drain, vocab)
    show_examples("Отсутствуют в словаре, но есть в последовательностях", missing_in_vocab, drain, vocab)
    show_examples("Лишние в словаре (нет в Drain)", orphan_in_vocab, drain, vocab)

    # --- Отчёт с расшифровкой шаблонов ---
    def format_event_with_template(eid: str) -> Dict[str, Any]:
        """Форматирует событие с шаблоном для отчета."""
        result: Dict[str, Any] = {"event_id": eid}
        template = drain.get(eid)
        if template:
            result["template"] = template
            result["source"] = "drain"
        else:
            vocab_entry = vocab.get(eid)
            if vocab_entry:
                result["vocab_entry"] = vocab_entry
                result["source"] = "vocab_only"
            else:
                result["template"] = "[не найден]"
                result["source"] = "none"
        return result
    
    report = {
        "stats": {
            "total_drain_templates": len(drain_ids),
            "total_vocab_entries": len(vocab_ids),
            "total_sequence_events": len(used),
        },
        "drain_unused": [format_event_with_template(eid) for eid in sorted(unused_in_sequences)],
        "missing_in_vocab": [format_event_with_template(eid) for eid in sorted(missing_in_vocab)],
        "orphan_in_vocab": [format_event_with_template(eid) for eid in sorted(orphan_in_vocab)],
    }

    # --- Определяем директорию для отчета ---
    # Используем директорию reports в корне проекта или рядом с drain_dir
    report_dir = Path("reports")
    if not report_dir.exists():
        # Попробуем создать reports рядом с drain_dir
        report_dir = drain_dir.parent / "reports"
    
    report_path = report_dir / "vocab_consistency_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Отчёт сохранён: {report_path.resolve()}")
    print("✅ Проверка завершена.")


if __name__ == "__main__":
    main()
