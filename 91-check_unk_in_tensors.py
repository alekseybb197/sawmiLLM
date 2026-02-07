#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
91-check_unk_in_tensors.py

Скрипт для проверки наличия [UNK] токенов в обучающих тензорах.

Проверяет:
1. Какие token_id реально встречаются в train.pt (input_ids и target_ids)
2. Есть ли [UNK] (unk_id) в обучающих данных
3. Статистику по использованию различных token_id
4. Сравнение с event_vocab.json

ИСПОЛЬЗОВАНИЕ:
==============
./91-check_unk_in_tensors.py
  - Автоматически берет пути из config.yaml:
    - train.pt: dataset.dataset/train.pt
    - vocab: dataset.vocab/event_vocab.json
    - val.pt: dataset.dataset/val.pt (опционально)

./91-check_unk_in_tensors.py --train-pt dataset/train.pt --vocab dataset/vocab/event_vocab.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import Counter

import torch

from modules.datasets import load_prepared_tensors
from modules.config import get_config, Config
from modules.logger import get_logger

logger = get_logger(__name__)


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], int, int]:
    """
    Загружает словарь из event_vocab.json.
    
    Returns:
        (event_to_id, pad_id, unk_id)
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    event_to_id = {}
    pad_id = 0
    unk_id = 1
    
    for event_id, token_id in data.items():
        if event_id == "[PAD]":
            pad_id = token_id
        elif event_id == "[UNK]":
            unk_id = token_id
        else:
            event_to_id[event_id] = token_id
    
    return event_to_id, pad_id, unk_id


def analyze_tensor(
    tensor: torch.Tensor,
    name: str,
    pad_id: int,
    unk_id: int,
    vocab_size: int,
) -> Dict:
    """
    Анализирует тензор и возвращает статистику.
    
    Args:
        tensor: Тензор с token_id
        name: Имя тензора (для логирования)
        pad_id: ID токена [PAD]
        unk_id: ID токена [UNK]
        vocab_size: Размер словаря
    
    Returns:
        Словарь со статистикой
    """
    # Преобразуем в numpy для анализа (если большой тензор, используем выборку)
    flat = tensor.flatten().cpu()
    
    # Подсчитываем уникальные значения
    unique_values, counts = torch.unique(flat, return_counts=True)
    unique_values = unique_values.tolist()
    counts = counts.tolist()
    
    # Создаем Counter для удобства
    token_counter = dict(zip(unique_values, counts))
    
    # Статистика
    total_tokens = flat.numel()
    pad_count = token_counter.get(pad_id, 0)
    unk_count = token_counter.get(unk_id, 0)
    
    # Минимальный и максимальный token_id
    min_token_id = min(unique_values)
    max_token_id = max(unique_values)
    
    # Проверяем, есть ли token_id за пределами словаря
    out_of_vocab = [tid for tid in unique_values if tid >= vocab_size]
    
    return {
        "name": name,
        "shape": list(tensor.shape),
        "total_tokens": total_tokens,
        "unique_token_ids": len(unique_values),
        "min_token_id": min_token_id,
        "max_token_id": max_token_id,
        "pad_count": pad_count,
        "pad_percent": (pad_count / total_tokens * 100) if total_tokens > 0 else 0.0,
        "unk_count": unk_count,
        "unk_percent": (unk_count / total_tokens * 100) if total_tokens > 0 else 0.0,
        "has_unk": unk_count > 0,
        "out_of_vocab": out_of_vocab,
        "token_counter": token_counter,
    }


def check_unk_in_tensors(
    train_pt: Path,
    vocab_path: Path,
    val_pt: Path | None = None,
) -> None:
    """
    Проверяет наличие [UNK] в обучающих тензорах.
    
    Args:
        train_pt: Путь к train.pt
        vocab_path: Путь к event_vocab.json
        val_pt: Путь к val.pt (опционально)
    """
    logger.info("=" * 80)
    logger.info("🔍 ПРОВЕРКА НАЛИЧИЯ [UNK] В ОБУЧАЮЩИХ ТЕНЗОРАХ")
    logger.info("=" * 80)
    
    # Загружаем словарь
    logger.info("📖 Загрузка словаря из %s", vocab_path)
    event_to_id, pad_id, unk_id = load_vocab(vocab_path)
    vocab_size = max(event_to_id.values()) + 1 if event_to_id else 2
    
    logger.info("   pad_id = %d", pad_id)
    logger.info("   unk_id = %d", unk_id)
    logger.info("   vocab_size = %d (включая [PAD] и [UNK])", vocab_size)
    logger.info("   уникальных событий = %d", len(event_to_id))
    
    # Загружаем train.pt
    logger.info("\n📦 Загрузка train.pt из %s", train_pt)
    train_data = load_prepared_tensors(train_pt)
    
    logger.info("   input_ids: %s", list(train_data.input_ids.shape))
    logger.info("   target_ids: %s", list(train_data.target_ids.shape))
    logger.info("   sec_ids: %s", list(train_data.sec_ids.shape))
    
    # Анализируем train тензоры
    logger.info("\n📊 АНАЛИЗ TRAIN ТЕНЗОРОВ")
    logger.info("-" * 80)
    
    input_stats = analyze_tensor(
        train_data.input_ids,
        "train.input_ids",
        pad_id,
        unk_id,
        vocab_size,
    )
    
    target_stats = analyze_tensor(
        train_data.target_ids,
        "train.target_ids",
        pad_id,
        unk_id,
        vocab_size,
    )
    
    logger.info("INPUT_IDS:")
    logger.info("   Всего токенов: %d", input_stats["total_tokens"])
    logger.info("   Уникальных token_id: %d", input_stats["unique_token_ids"])
    logger.info("   Диапазон token_id: [%d, %d]", input_stats["min_token_id"], input_stats["max_token_id"])
    logger.info("   [PAD] (%d): %d токенов (%.2f%%)", pad_id, input_stats["pad_count"], input_stats["pad_percent"])
    logger.info("   [UNK] (%d): %d токенов (%.2f%%)", unk_id, input_stats["unk_count"], input_stats["unk_percent"])
    logger.info("   Есть [UNK]: %s", "✅ ДА" if input_stats["has_unk"] else "❌ НЕТ")
    
    if input_stats["out_of_vocab"]:
        logger.warning("   ⚠️ Обнаружены token_id за пределами словаря: %s", input_stats["out_of_vocab"])
    
    logger.info("\nTARGET_IDS:")
    logger.info("   Всего токенов: %d", target_stats["total_tokens"])
    logger.info("   Уникальных token_id: %d", target_stats["unique_token_ids"])
    logger.info("   Диапазон token_id: [%d, %d]", target_stats["min_token_id"], target_stats["max_token_id"])
    logger.info("   [PAD] (%d): %d токенов (%.2f%%)", pad_id, target_stats["pad_count"], target_stats["pad_percent"])
    logger.info("   [UNK] (%d): %d токенов (%.2f%%)", unk_id, target_stats["unk_count"], target_stats["unk_percent"])
    logger.info("   Есть [UNK]: %s", "✅ ДА" if target_stats["has_unk"] else "❌ НЕТ")
    
    if target_stats["out_of_vocab"]:
        logger.warning("   ⚠️ Обнаружены token_id за пределами словаря: %s", target_stats["out_of_vocab"])
    
    # Топ-10 самых частых token_id (исключая PAD)
    logger.info("\n📈 ТОП-10 САМЫХ ЧАСТЫХ TOKEN_ID В TARGET_IDS (исключая PAD):")
    target_counter = target_stats["token_counter"]
    top_targets = sorted(
        [(tid, count) for tid, count in target_counter.items() if tid != pad_id],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for tid, count in top_targets:
        percent = (count / target_stats["total_tokens"] * 100) if target_stats["total_tokens"] > 0 else 0.0
        event_name = "[UNK]" if tid == unk_id else f"token_{tid}"
        logger.info("   token_id=%d (%s): %d раз (%.2f%%)", tid, event_name, count, percent)
    
    # Анализируем val.pt если указан
    if val_pt and val_pt.exists():
        logger.info("\n📦 Загрузка val.pt из %s", val_pt)
        val_data = load_prepared_tensors(val_pt)
        
        logger.info("\n📊 АНАЛИЗ VAL ТЕНЗОРОВ")
        logger.info("-" * 80)
        
        val_input_stats = analyze_tensor(
            val_data.input_ids,
            "val.input_ids",
            pad_id,
            unk_id,
            vocab_size,
        )
        
        val_target_stats = analyze_tensor(
            val_data.target_ids,
            "val.target_ids",
            pad_id,
            unk_id,
            vocab_size,
        )
        
        logger.info("VAL INPUT_IDS:")
        logger.info("   [UNK] (%d): %d токенов (%.2f%%)", unk_id, val_input_stats["unk_count"], val_input_stats["unk_percent"])
        logger.info("   Есть [UNK]: %s", "✅ ДА" if val_input_stats["has_unk"] else "❌ НЕТ")
        
        logger.info("\nVAL TARGET_IDS:")
        logger.info("   [UNK] (%d): %d токенов (%.2f%%)", unk_id, val_target_stats["unk_count"], val_target_stats["unk_percent"])
        logger.info("   Есть [UNK]: %s", "✅ ДА" if val_target_stats["has_unk"] else "❌ НЕТ")
    
    # Итоговый вывод
    logger.info("\n" + "=" * 80)
    logger.info("📋 ИТОГОВЫЙ ВЫВОД")
    logger.info("=" * 80)
    
    train_has_unk = input_stats["has_unk"] or target_stats["has_unk"]
    
    if train_has_unk:
        logger.warning("⚠️ В TRAIN ТЕНЗОРАХ ОБНАРУЖЕН [UNK]!")
        logger.warning("   Это означает, что в обучающих данных были события, которых нет в словаре.")
        logger.warning("   При инференсе новые паттерны также будут помечены как [UNK].")
    else:
        logger.info("✅ В TRAIN ТЕНЗОРАХ НЕТ [UNK]")
        logger.info("   Все события из обучающих данных присутствуют в словаре.")
        logger.info("   При инференсе любые новые паттерны будут помечены как [UNK] и должны считаться аномалиями.")
    
    logger.info("=" * 80)


def main() -> None:
    """
    Основная функция для проверки наличия [UNK] в тензорах.
    
    Использует настройки из config.yaml:
    - dataset.dataset: директория с train.pt и val.pt
    - dataset.vocab: директория со словарем event_vocab.json
    """
    parser = argparse.ArgumentParser(
        description="Проверка наличия [UNK] токенов в обучающих тензорах"
    )
    parser.add_argument(
        "--train-pt",
        type=str,
        help="Путь к train.pt (по умолчанию из config.yaml: dataset.dataset/train.pt)",
    )
    parser.add_argument(
        "--val-pt",
        type=str,
        help="Путь к val.pt (по умолчанию из config.yaml: dataset.dataset/val.pt, опционально)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="Путь к event_vocab.json (по умолчанию из config.yaml: dataset.vocab/event_vocab.json)",
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = get_config()
    
    # Определяем пути
    dataset_cfg = config.get("dataset", {})
    
    if args.train_pt:
        train_pt = Path(args.train_pt)
    else:
        dataset_dir_str = dataset_cfg.get("dataset")
        if not dataset_dir_str:
            raise ValueError(
                "Поле 'dataset.dataset' не указано в config.yaml. "
                "Пожалуйста, укажите путь к директории с train.pt или используйте --train-pt"
            )
        dataset_dir = Path(dataset_dir_str)
        train_pt = dataset_dir / "train.pt"
    
    if args.vocab:
        vocab_path = Path(args.vocab)
    else:
        vocab_dir_str = dataset_cfg.get("vocab")
        if not vocab_dir_str:
            raise ValueError(
                "Поле 'dataset.vocab' не указано в config.yaml. "
                "Пожалуйста, укажите путь к директории со словарем или используйте --vocab"
            )
        vocab_dir = Path(vocab_dir_str)
        vocab_path = vocab_dir / "event_vocab.json"
    
    val_pt = None
    if args.val_pt:
        val_pt = Path(args.val_pt)
    else:
        dataset_dir_str = dataset_cfg.get("dataset")
        if dataset_dir_str:
            dataset_dir = Path(dataset_dir_str)
            val_pt_path = dataset_dir / "val.pt"
            if val_pt_path.exists():
                val_pt = val_pt_path
    
    # Проверяем существование файлов
    if not train_pt.exists():
        raise FileNotFoundError(f"Файл train.pt не найден: {train_pt}")
    
    if not vocab_path.exists():
        raise FileNotFoundError(f"Файл event_vocab.json не найден: {vocab_path}")
    
    # Запускаем проверку
    check_unk_in_tensors(
        train_pt=train_pt,
        vocab_path=vocab_path,
        val_pt=val_pt,
    )


if __name__ == "__main__":
    main()

