#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
96-analyze_windowed_dataset.py — оценка эффективности подготовленного оконного датасета для LogBERT.

Читает настройки из config.yaml:
- dataset.dataset: директория с готовым датасетом (train.pt, val.pt, test.pt)
- dataset.vocab: путь к директории со словарем
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy

from modules.config import get_config, Config
from modules.logger import get_logger
from modules.utils import ensure_dir

log = get_logger(__name__)


def analyze_prepared_dataset(pt_path: Path, meta_path: Path, vocab_path: Path, outdir: Path):
    """Анализирует один подготовленный датасет."""
    data = torch.load(pt_path)
    input_ids = data["input_ids"]
    target_ids = data["target_ids"]
    sec_ids = data["sec_ids"]

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    n_windows = len(input_ids)
    window_size = input_ids.shape[1]
    vocab_size = len(vocab)

    # частоты событий
    all_events = input_ids.flatten().tolist() + target_ids.tolist()
    freq = Counter(all_events)
    uniq_events = len(freq)
    top10 = freq.most_common(10)

    # покрытие словаря
    coverage = uniq_events / vocab_size

    # частоты секций
    sec_freq = Counter(sec_ids.tolist())
    sec_entropy = entropy(list(sec_freq.values()))

    # оценка разнообразия событий в окне
    uniq_per_win = [len(set(win.tolist())) for win in input_ids[:min(5000, n_windows)]]
    avg_uniq_per_win = float(np.mean(uniq_per_win))

    split_name = pt_path.stem  # "train", "val", "test"
    print(f"""📊 Prepared dataset analysis
File: {pt_path.name}
Split: {split_name}
---------------------------------------
Windows:             {n_windows:,}
Window size:         {window_size}
Unique events:       {uniq_events:,}/{vocab_size:,} ({coverage:.1%})
Avg unique per win:  {avg_uniq_per_win:.2f}
Section entropy:     {sec_entropy:.3f}
Top-10 events:       {[inv_vocab[i] for i, _ in top10]}
---------------------------------------
""")

    # Визуализация
    split_outdir = outdir / split_name
    ensure_dir(split_outdir)

    plt.figure(figsize=(8, 5))
    ids, counts = zip(*top10)
    names = [inv_vocab[i] for i in ids]
    plt.bar(names, counts, color='steelblue')
    plt.title(f'Top-10 frequent events ({split_name})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(split_outdir / "top10_events.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(uniq_per_win, bins=30, color='orange', alpha=0.7)
    plt.title(f"Distribution of unique events per window ({split_name})")
    plt.xlabel("unique events")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(split_outdir / "unique_per_window.png", dpi=200)
    plt.close()

    print(f"✅ Saved plots to {split_outdir}")


def _get_dataset_dir_from_config(cfg: Config) -> Path | None:
    """Получает директорию с датасетом из dataset.dataset."""
    dataset_dir = cfg.get("dataset.dataset")
    return Path(dataset_dir) if dataset_dir else None


def _get_vocab_path_from_config(cfg: Config) -> Path | None:
    """Получает путь к словарю из dataset.vocab/event_vocab.json."""
    vocab_dir = cfg.get("dataset.vocab")
    if not vocab_dir:
        return None
    vocab_path = Path(vocab_dir) / "event_vocab.json"
    return vocab_path if vocab_path.exists() else None


def _find_dataset_splits(dataset_dir: Path) -> List[str]:
    """Находит доступные split'ы (train, val, test) в директории датасета."""
    splits = []
    for split_name in ["train", "val", "test"]:
        pt_path = dataset_dir / f"{split_name}.pt"
        if pt_path.exists():
            splits.append(split_name)
    return splits


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Analyze prepared LogBERT dataset from config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Пример использования:
  python analyze_windowed_dataset.py                    # Анализирует все split'ы из config.yaml
  python analyze_windowed_dataset.py --split train     # Анализирует только train
  python analyze_windowed_dataset.py --pt file.pt     # Анализирует конкретный файл
        """
    )
    ap.add_argument("--pt", type=Path, default=None,
                    help="Path to .pt file (overrides config and --split)")
    ap.add_argument("--meta", type=Path, default=None,
                    help="Path to meta.jsonl (overrides config)")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="Path to event_vocab.json (overrides dataset.vocab)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output dir for report/plots (default: dataset_dir/analysis)")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default=None,
                    help="Analyze specific split (train/val/test)")
    args = ap.parse_args()

    # Загружаем конфиг
    cfg = get_config()

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

    # Если указан конкретный .pt файл, анализируем его
    if args.pt:
        if not args.pt.exists():
            ap.error(f"PT file not found: {args.pt}")
        
        # Определяем соответствующий meta файл
        meta_path = args.meta
        if meta_path is None:
            # Пытаемся найти meta файл рядом с .pt
            meta_path = args.pt.parent / f"{args.pt.stem}_meta.jsonl"
            if not meta_path.exists():
                ap.error(f"Meta file not found: {meta_path}. Specify --meta")
        
        # Определяем директорию для вывода
        outdir = args.outdir
        if outdir is None:
            outdir = args.pt.parent / "analysis"
        ensure_dir(outdir)
        
        analyze_prepared_dataset(args.pt, meta_path, vocab_path, outdir)
        return

    # Иначе работаем с датасетом из конфига
    dataset_dir = _get_dataset_dir_from_config(cfg)
    if dataset_dir is None:
        ap.error("Dataset directory not specified. Set --pt or configure dataset.dataset in config.yaml")
    
    if not dataset_dir.exists():
        ap.error(f"Dataset directory not found: {dataset_dir}")

    # Находим доступные split'ы
    available_splits = _find_dataset_splits(dataset_dir)
    if not available_splits:
        ap.error(f"No dataset files found in {dataset_dir}. Expected train.pt, val.pt, or test.pt")

    # Определяем какие split'ы анализировать
    splits_to_analyze = [args.split] if args.split else available_splits
    if args.split and args.split not in available_splits:
        ap.error(f"Split '{args.split}' not found. Available: {available_splits}")

    # Определяем директорию для вывода
    outdir = args.outdir
    if outdir is None:
        outdir = dataset_dir / "analysis"
    ensure_dir(outdir)

    log.info(f"Analyzing splits: {splits_to_analyze}")
    
    # Анализируем каждый split
    for split_name in splits_to_analyze:
        pt_path = dataset_dir / f"{split_name}.pt"
        meta_path = dataset_dir / f"{split_name}_meta.jsonl"
        
        if not pt_path.exists():
            log.warning(f"Skipping {split_name}: {pt_path} not found")
            continue
        
        if not meta_path.exists():
            log.warning(f"Meta file not found for {split_name}: {meta_path}")
        
        log.info(f"Analyzing {split_name}...")
        analyze_prepared_dataset(pt_path, meta_path, vocab_path, outdir)
    
    log.info(f"✅ Analysis complete. Results saved to {outdir}")


if __name__ == "__main__":
    main()
