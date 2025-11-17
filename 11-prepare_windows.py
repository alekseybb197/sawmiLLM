#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_prepare_windows.py — CLI для нарезки оконных выборок.

Читает настройки из config.yaml:
- dataset.drain_list: входные JSONL файлы (train.jsonl, val.jsonl, test.jsonl)
- dataset.vocab: путь к директории со словарем
- dataset.windows.size: размер окна
- dataset.dataset: директория для сохранения готового датасета
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import argparse
import json

from modules.windowing import prepare_windows
from modules.config import get_config, Config
from modules.logger import get_logger
from modules.utils import ensure_dir

log = get_logger(__name__)


def _get_inputs_from_config(cfg: Config) -> List[Path]:
    """Получает список входных файлов из dataset.drain_list."""
    drain_list = cfg.get("dataset.drain_list", [])
    if not drain_list:
        return []
    
    inputs = []
    for path_str in drain_list:
        path = Path(path_str)
        if path.exists():
            inputs.append(path)
        else:
            log.warning(f"Input file not found: {path}")
    return inputs


def _get_vocab_path_from_config(cfg: Config) -> Path | None:
    """Получает путь к словарю из dataset.vocab/event_vocab.json."""
    vocab_dir = cfg.get("dataset.vocab")
    if not vocab_dir:
        return None
    vocab_path = Path(vocab_dir) / "event_vocab.json"
    return vocab_path if vocab_path.exists() else None


def _get_output_dir_from_config(cfg: Config) -> Path | None:
    """Получает директорию для вывода из dataset.dataset."""
    dataset_dir = cfg.get("dataset.dataset")
    return Path(dataset_dir) if dataset_dir else None


def _get_window_size_from_config(cfg: Config) -> int:
    """Получает размер окна из dataset.windows.size."""
    window_size = cfg.get("dataset.windows.size", 64)
    return int(window_size)


def main():
    ap = argparse.ArgumentParser(
        description='Prepare windows for LogBERT from config.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Пример использования:
  python 02_prepare_windows.py                    # Использует все настройки из config.yaml
  python 02_prepare_windows.py --window-size 32    # Переопределяет размер окна
  python 02_prepare_windows.py --inputs file.jsonl # Переопределяет входные файлы
        """
    )
    ap.add_argument('--inputs', nargs='+', type=Path, default=None,
                    help='Input JSONL files (overrides dataset.drain_list)')
    ap.add_argument('--vocab', type=Path, default=None,
                    help='Path to vocab file (overrides dataset.vocab/event_vocab.json)')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='Output directory (overrides dataset.dataset)')
    ap.add_argument('--window-size', type=int, default=None,
                    help='Window size (overrides dataset.windows.size)')
    ap.add_argument('--stride', type=int, default=1,
                    help='Stride for windowing (default: 1)')
    ap.add_argument('--rle-cap', type=int, default=5,
                    help='RLE cap for event sequences (default: 5)')
    ap.add_argument('--section-buckets', type=int, default=2048,
                    help='Number of section buckets (default: 2048)')
    ap.add_argument('--accept-status', type=str, default=None,
                    help='Filter by status (e.g., "success", "error")')
    ap.add_argument('--config', type=Path, default=Path("config.yaml"),
                    help='Path to config.yaml file')
    args = ap.parse_args()

    # Загружаем конфиг
    cfg = get_config()

    # Определяем входные файлы
    inputs = list(args.inputs) if args.inputs else []
    if not inputs:
        inputs = _get_inputs_from_config(cfg)
        if inputs:
            log.info(f"Inputs from config: {[str(p) for p in inputs]}")
    
    if not inputs:
        ap.error("No inputs provided. Set --inputs or configure dataset.drain_list in config.yaml")

    # Определяем путь к словарю
    vocab_path = args.vocab
    if vocab_path is None:
        vocab_path_cfg = _get_vocab_path_from_config(cfg)
        if vocab_path_cfg:
            vocab_path = vocab_path_cfg
            log.info(f"Vocab path from config: {vocab_path}")
        else:
            ap.error("Vocab path not specified. Set --vocab or configure dataset.vocab in config.yaml")
    
    if not vocab_path.exists():
        ap.error(f"Vocab file not found: {vocab_path}")

    # Определяем директорию для вывода
    out_dir = args.out_dir
    if out_dir is None:
        out_dir_cfg = _get_output_dir_from_config(cfg)
        if out_dir_cfg:
            out_dir = out_dir_cfg
            log.info(f"Output dir from config: {out_dir}")
        else:
            ap.error("Output directory not specified. Set --out-dir or configure dataset.dataset in config.yaml")
    
    ensure_dir(out_dir)

    # Определяем размер окна
    window_size = args.window_size
    if window_size is None:
        window_size = _get_window_size_from_config(cfg)
        log.info(f"Window size from config: {window_size}")

    # Загружаем словарь
    with vocab_path.open('r', encoding='utf-8') as f:
        vocab = json.load(f)
    log.info(f"Loaded vocab with {len(vocab)} tokens")

    # Обрабатываем каждый входной файл отдельно
    for input_file in inputs:
        # Определяем split из имени файла (train, val, test)
        split_name = input_file.stem  # "train", "val", "test"
        if split_name not in ["train", "val", "test"]:
            # Пытаемся извлечь из пути, например "train.jsonl" -> "train"
            log.warning(f"Unexpected input file name: {input_file}, using stem as split name")
        
        # Формируем пути для вывода
        out_pt = out_dir / f"{split_name}.pt"
        out_meta = out_dir / f"{split_name}_meta.jsonl"
        
        log.info(f"Processing {input_file} -> {out_pt}")
        prepare_windows(
            [input_file],
            vocab,
            out_pt,
            out_meta,
            window_size=window_size,
            stride=args.stride,
            rle_cap_n=args.rle_cap,
            section_buckets=args.section_buckets,
            accept_status=args.accept_status
        )
    
    log.info('Done prepare_windows')


if __name__ == '__main__':
    main()
