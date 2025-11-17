#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_vocab.py — CLI для построения словаря событий Drain и разделения датасета.

Особенности:
  - Поддерживает чтение параметров из config.yaml:
      • dataset.vocab_list / dataset.dataset_list / dataset.prepared — входные файлы
      • dataset.vocab — выходная директория для словаря
      • dataset.drain — выходная директория для train/val/test.jsonl
      • dataset.train_split / val_split / random_seed — параметры разделения
      • vocab.min_support — минимальная частота включения событий
      • vocab.exclude_patterns — regex-паттерны для исключения
  - Использует modules/vocab.build_event_vocab()
  - Функционал разделения датасета перенесен из 03_drain_dataset.py
  - Всегда выполняет сначала разделение датасета, затем построение словаря

Примеры:
  # Полный цикл: разделение датасета + построение словаря:
  python 04_build_vocab.py
  python 04_build_vocab.py --config config.yaml
  python 04_build_vocab.py --inputs dataset/drain/drain_chunks.jsonl
"""
from __future__ import annotations
from pathlib import Path
from typing import cast, Dict, Any, List, Tuple, Optional
import json
import random
import hashlib
import yaml
import typer

from modules.logger import get_logger
from modules.utils import ensure_dir
from modules.vocab import build_event_vocab

log = get_logger(__name__)


def load_config(path: Path) -> Dict[Any, Any]:
    """Загружает конфиг напрямую из YAML файла."""
    with path.open("r", encoding="utf-8") as f:
        return cast(Dict[Any, Any], yaml.safe_load(f))


def _inputs_from_config(cfg: dict, for_split: bool = False) -> List[Path]:
    """
    Извлекает входные JSONL файлы из конфига.
    
    Args:
        cfg: Конфигурация из config.yaml
        for_split: Если True, использует dataset.drain_chunks для разделения датасета
                   Если False, использует vocab_list для построения словаря
    
    Returns:
        Список путей к входным файлам
    """
    ds = cfg.get("dataset", {})
    out: List[Path] = []

    if for_split:
        # Для разделения датасета используем dataset.drain_chunks
        drain_chunks = ds.get("drain_chunks")
        if isinstance(drain_chunks, str):
            p = Path(drain_chunks)
            if p.exists():
                out.append(p)
            else:
                log.warning(f"[Split] drain_chunks файл не найден: {p}")
        else:
            log.warning("[Split] dataset.drain_chunks не указан в config.yaml")
    else:
        # Для построения словаря используем vocab_list или dataset_list
        # 1. dataset.vocab_list (приоритетный)
        for key in ["vocab_list", "dataset_list"]:
            lst = ds.get(key, [])
            if isinstance(lst, list):
                for item in lst:
                    p = Path(item)
                    if p.exists():
                        out.append(p)
                    else:
                        log.warning(f"[Vocab] Config file not found: {p}")

        # 2. Если списки пусты — dataset.prepared (директория)
        if not out and isinstance(ds.get("prepared"), str):
            p = Path(ds["prepared"])
            if p.exists() and p.is_dir():
                out.append(p)

    return out


def _vocab_dir_from_config(cfg: dict) -> Path | None:
    """Извлекает путь для выходного словаря из dataset.vocab."""
    ds = cfg.get("dataset", {})
    vocab_path = ds.get("vocab")
    if isinstance(vocab_path, str):
        return Path(vocab_path)
    return None


# ---------------------------------------------------------
# Split & save functions (перенесены из 03_drain_dataset.py)
# ---------------------------------------------------------
def split_by_build_id(records: List[dict], train_ratio: float, val_ratio: float, seed: int) -> tuple[List[dict], List[dict], List[dict]]:
    """
    Разделяет записи на train/val/test по группам (pipeline_id, section_name).
    
    Гибридный подход:
    1. Группирует записи по (pipeline_id, section_name)
    2. Распределяет ГРУППЫ целиком по train/val/test согласно пропорциям
    3. Гарантирует, что все группы из val/test также присутствуют в train
       (для исключения уникальных групп в val/test)
    
    Преимущества:
    - Строгое разделение: группы не пересекаются между val и test
    - Гарантия присутствия: все группы из val/test есть в train
    - Простота и детерминированность
    - Меньше вероятность data leakage между val/test
    
    Args:
        records: Список записей с полями pipeline_id, build_id, section_name, event_seq
        train_ratio: Доля ГРУПП для train (например, 0.8)
        val_ratio: Доля ГРУПП для val (например, 0.1)
        seed: Seed для случайного перемешивания
    
    Returns:
        tuple: (train_records, val_records, test_records)
        Примечание: train будет содержать больше чанков, чем указано в train_ratio,
        так как включает все группы из val/test для гарантии присутствия.
    """
    rnd = random.Random(seed)
    
    # Вычисляем test_ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Группируем по (pipeline_id, section_name)
    by_group: Dict[Tuple[str, str], List[dict]] = {}
    for r in records:
        pipeline_id = r.get("pipeline_id", "unknown")
        section_name = r.get("section_name", "full")
        key = (pipeline_id, section_name)
        by_group.setdefault(key, []).append(r)
    
    # Перемешиваем группы детерминированно
    group_keys = list(by_group.keys())
    rnd.shuffle(group_keys)
    
    # Распределяем ГРУППЫ по train/val/test согласно пропорциям
    n_groups = len(group_keys)
    n_train_groups = int(n_groups * train_ratio)
    n_val_groups = int(n_groups * val_ratio)
    n_test_groups = n_groups - n_train_groups - n_val_groups  # остаток идёт в test
    
    train_group_keys = group_keys[:n_train_groups]
    val_group_keys = group_keys[n_train_groups:n_train_groups + n_val_groups]
    test_group_keys = group_keys[n_train_groups + n_val_groups:]
    
    # Собираем чанки
    train: List[dict] = []
    val: List[dict] = []
    test: List[dict] = []
    
    # 1. Группы только для train
    for key in train_group_keys:
        train.extend(by_group[key])
    
    # 2. Группы для val (добавляем в val И в train для гарантии присутствия)
    for key in val_group_keys:
        chunks = by_group[key]
        val.extend(chunks)
        train.extend(chunks)  # Также в train для гарантии
    
    # 3. Группы для test (добавляем в test И в train для гарантии присутствия)
    for key in test_group_keys:
        chunks = by_group[key]
        test.extend(chunks)
        train.extend(chunks)  # Также в train для гарантии
    
    return train, val, test


def save_split_files(train_file: Path, val_file: Path, test_file: Path, 
                     train: List[dict], val: List[dict], test: List[dict], 
                     rewrite: bool, logger) -> None:
    """
    Сохраняет разделенные данные в файлы train/val/test.jsonl.
    
    Args:
        train_file: Путь к файлу train.jsonl
        val_file: Путь к файлу val.jsonl
        test_file: Путь к файлу test.jsonl
        train: Список записей для train
        val: Список записей для val
        test: Список записей для test
        rewrite: Если True, перезаписывает файлы, иначе дописывает
        logger: Логгер для вывода сообщений
    """
    mode = "w" if rewrite else "a"
    for name, data, fp in [("train", train, train_file), ("val", val, val_file), ("test", test, test_file)]:
        with open(fp, mode, encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"{name}: {len(data)} записей ({'rewrite' if rewrite else 'append'})")


def print_split_report(train: List[dict], val: List[dict], test: List[dict], 
                       train_file: Path, val_file: Path, test_file: Path, 
                       logger) -> None:
    """
    Выводит отчет о разделении датасета: количество чанков, размеры файлов, статистику по группам.
    
    Args:
        train: Список записей для train
        val: Список записей для val
        test: Список записей для test
        train_file: Путь к файлу train.jsonl
        val_file: Путь к файлу val.jsonl
        test_file: Путь к файлу test.jsonl
        logger: Логгер для вывода сообщений
    """
    total = len(train) + len(val) + len(test)
    
    # Размеры файлов
    def get_file_size(file_path: Path) -> str:
        if file_path.exists():
            size_bytes = file_path.stat().st_size
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.2f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.2f} MB"
        return "N/A"
    
    train_size = get_file_size(train_file)
    val_size = get_file_size(val_file)
    test_size = get_file_size(test_file)
    
    # Доли
    train_ratio = len(train) / total if total > 0 else 0
    val_ratio = len(val) / total if total > 0 else 0
    test_ratio = len(test) / total if total > 0 else 0
    
    # Статистика по группам (pipeline_id, section_name)
    def get_group_key(rec: dict) -> Tuple[str, str]:
        return (
            rec.get("pipeline_id", "unknown"),
            rec.get("section_name", "full")
        )
    
    def count_by_split(records: List[dict]) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for rec in records:
            key = get_group_key(rec)
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    train_groups = count_by_split(train)
    val_groups = count_by_split(val)
    test_groups = count_by_split(test)
    
    # Все уникальные группы
    all_groups = set(train_groups.keys()) | set(val_groups.keys()) | set(test_groups.keys())
    
    # Группы, которые есть в нескольких сплитах
    groups_in_multiple_splits = []
    for group in all_groups:
        in_train = group in train_groups
        in_val = group in val_groups
        in_test = group in test_groups
        if sum([in_train, in_val, in_test]) > 1:
            groups_in_multiple_splits.append(group)
    
    # Выводим отчет
    logger.info("=" * 80)
    logger.info("📊 ОТЧЕТ О РАЗДЕЛЕНИИ ДАТАСЕТА (Гибридный подход)")
    logger.info("=" * 80)
    logger.info("ℹ️  Примечание: Train содержит больше чанков, так как включает все группы из val/test")
    logger.info("   для гарантии отсутствия уникальных групп в val/test.")
    
    logger.info(f"\n📦 Общая статистика:")
    logger.info(f"  Всего чанков: {total:,}")
    logger.info(f"  Train: {len(train):,} чанков ({train_ratio:.1%}) | Размер: {train_size}")
    logger.info(f"  Val:   {len(val):,} чанков ({val_ratio:.1%}) | Размер: {val_size}")
    logger.info(f"  Test:  {len(test):,} чанков ({test_ratio:.1%}) | Размер: {test_size}")
    
    logger.info(f"\n📋 Статистика по группам (pipeline_id, section_name):")
    logger.info(f"  Всего уникальных групп: {len(all_groups):,}")
    logger.info(f"  Групп в train: {len(train_groups):,}")
    logger.info(f"  Групп в val:   {len(val_groups):,}")
    logger.info(f"  Групп в test:  {len(test_groups):,}")
    logger.info(f"  Групп в нескольких сплитах: {len(groups_in_multiple_splits):,}")
    
    # Топ-10 групп по количеству чанков
    all_group_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for group in all_groups:
        all_group_counts[group] = {
            "train": train_groups.get(group, 0),
            "val": val_groups.get(group, 0),
            "test": test_groups.get(group, 0),
            "total": train_groups.get(group, 0) + val_groups.get(group, 0) + test_groups.get(group, 0)
        }
    
    # Сортируем по общему количеству
    sorted_groups = sorted(all_group_counts.items(), key=lambda x: x[1]["total"], reverse=True)
    
    logger.info(f"\n🔝 Топ-10 групп по количеству чанков:")
    for i, (group, counts) in enumerate(sorted_groups[:10], 1):
        pipeline_id, section_name = group
        logger.info(
            f"  {i:2d}. {section_name[:40]:40s} | "
            f"pipeline={pipeline_id} | "
            f"train={counts['train']:3d} val={counts['val']:3d} test={counts['test']:3d} | "
            f"total={counts['total']:3d}"
        )
    
    # Группы, которые есть только в train (не должны быть в val/test согласно требованиям)
    groups_only_in_train = set(train_groups.keys()) - set(val_groups.keys()) - set(test_groups.keys())
    groups_in_val_or_test = set(val_groups.keys()) | set(test_groups.keys())
    groups_in_train_and_others = set(train_groups.keys()) & groups_in_val_or_test
    
    logger.info(f"\n✅ Качество разделения:")
    logger.info(f"  Групп только в train: {len(groups_only_in_train):,}")
    logger.info(f"  Групп в val/test: {len(groups_in_val_or_test):,}")
    logger.info(f"  Групп в train И val/test: {len(groups_in_train_and_others):,}")
    logger.info(f"  ✓ Все группы из val/test также присутствуют в train: {len(groups_in_val_or_test - groups_in_train_and_others) == 0}")
    
    logger.info("=" * 80)


def assign_split_and_append(train_file: Path, val_file: Path, test_file: Path,
                           records: List[dict], train_split: float, val_split: float,
                           logger) -> None:
    """
    Детерминированно назначает записи в train/val/test по хешу build_id.
    Используется для инкрементального добавления данных.
    
    Args:
        train_file: Путь к файлу train.jsonl
        val_file: Путь к файлу val.jsonl
        test_file: Путь к файлу test.jsonl
        records: Список записей для добавления
        train_split: Доля для train (например, 0.8)
        val_split: Доля для val (например, 0.1)
        logger: Логгер для вывода сообщений
    """
    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        build_id = r["build_id"]
        h = int(hashlib.sha1(build_id.encode()).hexdigest()[:8], 16) % 100
        train_thr = int(train_split * 100)
        val_thr = int((train_split + val_split) * 100)
        split = "train" if h < train_thr else "val" if h < val_thr else "test"
        by_split[split].append(r)
    save_split_files(train_file, val_file, test_file, 
                     by_split["train"], by_split["val"], by_split["test"], 
                     rewrite=False, logger=logger)


# Обертки для совместимости с 03_drain_dataset.py (используют cfg вместо отдельных путей)
def save_split(cfg: dict, train: List[dict], val: List[dict], test: List[dict], rewrite: bool, logger) -> None:
    """
    Обертка для save_split_files, использует cfg для путей к файлам.
    Используется для совместимости с 03_drain_dataset.py.
    """
    save_split_files(cfg["train_file"], cfg["val_file"], cfg["test_file"], train, val, test, rewrite, logger)


def assign_split_and_append_from_cfg(cfg: dict, records: List[dict], logger) -> None:
    """
    Обертка для assign_split_and_append, использует cfg для путей и параметров.
    Используется для совместимости с 03_drain_dataset.py.
    """
    assign_split_and_append(
        cfg["train_file"], cfg["val_file"], cfg["test_file"],
        records, cfg["train_split"], cfg["val_split"], logger
    )


def split_dataset(inputs: List[Path], output_dir: Path, cfg: dict, logger, append_mode: bool = False) -> None:
    """
    Разделяет JSONL файлы с чанками на train/val/test.
    
    Args:
        inputs: Список путей к JSONL файлам с записями (pipeline_id, build_id, event_seq, ...)
        output_dir: Директория для сохранения train/val/test.jsonl
        cfg: Конфигурация из config.yaml
        logger: Логгер
        append_mode: Если True, дописывает в существующие файлы с детерминированным назначением
    """
    dataset_cfg = cfg.get("dataset", {})
    
    # Параметры разделения
    train_split = float(dataset_cfg.get("train_split", 0.8))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    random_seed = int(dataset_cfg.get("random_seed", 42))
    
    # Пути к выходным файлам
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"
    
    # Загружаем все записи из входных файлов
    all_records = []
    for input_path in inputs:
        if input_path.is_dir():
            # Если директория, ищем все .jsonl файлы
            jsonl_files = list(input_path.glob("*.jsonl"))
            if not jsonl_files:
                logger.warning(f"[Split] No JSONL files found in {input_path}")
                continue
            for jsonl_file in jsonl_files:
                logger.info(f"[Split] Reading {jsonl_file}")
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            # Проверяем наличие обязательных полей
                            if "build_id" in rec and "event_seq" in rec:
                                all_records.append(rec)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"[Split] Skipping invalid line: {e}")
        elif input_path.is_file():
            logger.info(f"[Split] Reading {input_path}")
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if "build_id" in rec and "event_seq" in rec:
                            all_records.append(rec)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"[Split] Skipping invalid line: {e}")
    
    if not all_records:
        logger.error("[Split] No valid records found in input files")
        return
    
    logger.info(f"[Split] Loaded {len(all_records)} records from {len(inputs)} input(s)")
    
    # Разделяем записи
    if append_mode:
        logger.info(f"[Split] Append mode: deterministically assigning records to splits")
        assign_split_and_append(train_file, val_file, test_file, all_records, 
                               train_split, val_split, logger)
    else:
        logger.info(f"[Split] Split mode (hybrid approach): train={train_split:.1%}, val={val_split:.1%}, test={1-train_split-val_split:.1%}, seed={random_seed}")
        logger.info(f"[Split] Groups will be distributed entirely (no overlap between val/test)")
        logger.info(f"[Split] Groups from val/test will also be added to train for guarantee")
        train, val, test = split_by_build_id(all_records, train_split, val_split, random_seed)
        save_split_files(train_file, val_file, test_file, train, val, test, rewrite=True, logger=logger)
        
        # Выводим отчет о разделении
        print_split_report(train, val, test, train_file, val_file, test_file, logger)
    
    logger.info(f"[Split] Done. Output files: {train_file}, {val_file}, {test_file}")


# ---------------------------------------------------------
# CLI (Typer)
# ---------------------------------------------------------
app = typer.Typer(
    name="build_vocab",
    help="Построение словаря событий Drain и разделение датасета на train/val/test",
    add_completion=False,
)


@app.command()
def main(
    inputs: Optional[List[Path]] = typer.Option(None, "--inputs", "-i", help="JSONL файлы или директории (опционально, если указан config)"),
    outdir: Optional[Path] = typer.Option(None, "--outdir", "-o", help="Выходная директория для словаря (по умолчанию из config.yaml)"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Путь к файлу config.yaml"),
    split_output: Optional[Path] = typer.Option(None, "--split-output", help="Выходная директория для split файлов (по умолчанию: dataset.drain)"),
    split_append: bool = typer.Option(False, "--split-append", help="Режим добавления для split (детерминированное назначение)"),
) -> None:

    """
    Построение словаря событий Drain и разделение датасета на train/val/test.
    
    Всегда выполняет сначала разделение датасета, затем построение словаря.
    
    Примеры:
      # Полный цикл: разделение датасета + построение словаря:
      python 04_build_vocab.py
      python 04_build_vocab.py --config config.yaml
      python 04_build_vocab.py --inputs dataset/drain/drain_chunks.jsonl
    """
    # === Загружаем config.yaml ===
    cfg: dict | None = None
    if config.exists():
        cfg = load_config(config)
        log.info(f"[Vocab] Loaded config from {config}")
    else:
        log.warning(f"[Vocab] Config not found at {config}, using defaults")

    if cfg is None:
        typer.echo("❌ Ошибка: требуется наличие config.yaml", err=True)
        raise typer.Exit(code=1)

    # === Шаг 1: Разделение датасета ===
    log.info("=== Шаг 1: Разделение датасета ===")
    
    # Определяем входные файлы для split (используем dataset.drain_chunks)
    split_input_files = list(inputs) if inputs else []
    if not split_input_files and cfg:
        cfg_split_inputs = _inputs_from_config(cfg, for_split=True)
        if cfg_split_inputs:
            split_input_files.extend(cfg_split_inputs)
            log.info(f"[Split] Inputs from config (drain_chunks): {[str(p) for p in cfg_split_inputs]}")
    
    if not split_input_files:
        typer.echo("❌ Ошибка: для разделения датасета необходимо указать входные файлы (--inputs) или настроить dataset.drain_chunks в config.yaml", err=True)
        raise typer.Exit(code=1)
    
    # Определяем выходную директорию для split
    split_output_dir = split_output
    if split_output_dir is None:
        dataset_cfg = cfg.get("dataset", {})
        drain_dir = dataset_cfg.get("drain")
        if drain_dir:
            split_output_dir = Path(drain_dir)
        else:
            typer.echo("❌ Ошибка: --split-output не указан и dataset.drain не найден в config.yaml", err=True)
            raise typer.Exit(code=1)
    
    split_dataset(split_input_files, split_output_dir, cfg, log, append_mode=split_append)
    log.info("✅ Разделение датасета завершено")
    
    # === Шаг 2: Построение словаря ===
    log.info("=== Шаг 2: Построение словаря ===")
    
    # Определяем входные файлы для vocab (используем vocab_list или те же что указаны в --inputs)
    vocab_input_files = list(inputs) if inputs else []
    if not vocab_input_files and cfg:
        cfg_vocab_inputs = _inputs_from_config(cfg, for_split=False)
        if cfg_vocab_inputs:
            vocab_input_files.extend(cfg_vocab_inputs)
            log.info(f"[Vocab] Inputs from config (vocab_list): {[str(p) for p in cfg_vocab_inputs]}")
    
    if not vocab_input_files:
        typer.echo("❌ Ошибка: для построения словаря необходимо указать входные файлы (--inputs) или настроить dataset.vocab_list в config.yaml", err=True)
        raise typer.Exit(code=1)
    
    # Определяем выходную директорию для словаря
    vocab_output_dir = outdir
    if vocab_output_dir is None:
        vocab_dir = _vocab_dir_from_config(cfg)
        if vocab_dir:
            vocab_output_dir = vocab_dir
            log.info(f"[Vocab] Output dir from config: {vocab_output_dir}")
    
    if vocab_output_dir is None:
        typer.echo("❌ Ошибка: --outdir не указан и dataset.vocab не найден в config.yaml", err=True)
        raise typer.Exit(code=1)

    ensure_dir(vocab_output_dir)

    # Читаем параметры фильтрации из config.yaml
    min_support = 1
    exclude_patterns: list[str] | None = None
    drain_templates_path: Path | None = None

    vocab_cfg = cfg.get("vocab", {})
    min_support = int(vocab_cfg.get("min_support", 1))
    ep = vocab_cfg.get("exclude_patterns")
    if isinstance(ep, list) and all(isinstance(x, str) for x in ep):
        exclude_patterns = ep
    
    # Определяем путь к drain_templates.json из dataset.drain
    dataset_cfg = cfg.get("dataset", {})
    drain_dir = dataset_cfg.get("drain")
    if drain_dir:
        drain_templates_path = Path(drain_dir) / "drain_templates.json"
    else:
        # Fallback: пробуем найти в текущей директории
        drain_templates_path = Path("drain_templates.json")
    
    # Если есть явное указание в vocab.drain_templates, используем его
    dt = vocab_cfg.get("drain_templates")
    if isinstance(dt, str):
        drain_templates_path = Path(dt)
    
    log.info(
        f"[Vocab] Config params: min_support={min_support}, "
        f"exclude_patterns={len(exclude_patterns or [])}, "
        f"drain_templates={drain_templates_path}"
    )
    
    vocab_path, stats_path = build_event_vocab(
        inputs=vocab_input_files,
        outdir=vocab_output_dir,
        drain_templates_path=drain_templates_path,
        min_support=min_support,
        exclude_patterns=exclude_patterns,
    )
    log.info(f"✅ Построение словаря завершено. Saved vocab={vocab_path}, stats={stats_path}")
    
    log.info("✅ Все операции завершены успешно!")


if __name__ == "__main__":
    app()
