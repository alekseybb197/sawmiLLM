#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_calibrate_threshold.py — калибровка порога NLL по success-валидации.

ЦЕЛЬ:
=====
Вычисляет порог NLL (Negative Log-Likelihood) для детектирования аномалий в логах.

ПРИНЦИП РАБОТЫ:
===============
1. Загружает обученную модель LogBERT и валидационные данные (только success-логи)
2. Вычисляет NLL для каждого примера в валидационном наборе
3. Находит процентиль (например, 99.5%) от распределения NLL значений
4. Сохраняет этот порог в thresholds.json

ИСПОЛЬЗОВАНИЕ ПОРОГА:
====================
При детектировании аномалий:
- Если NLL нового лога > threshold → вероятно аномалия
- Если NLL нового лога <= threshold → вероятно нормальный лог

Почему success-логи?
- Мы используем только success-логи для калибровки, чтобы порог отражал
  нормальное поведение системы
- Если порог вычислен на success-логах, то аномальные (failed) логи будут
  иметь более высокий NLL и будут детектироваться

ПАРАМЕТРЫ:
==========
Все параметры берутся из config.yaml (см. функцию main()).

ИСПОЛЬЗОВАНИЕ:
=============
python 13-calibrate_threshold.py
"""
from __future__ import annotations
from pathlib import Path
import json, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.model import UnifiedLogBERT
from modules.datasets import WindowsDataset, load_prepared_tensors
from modules.config import get_config
from modules.logger import get_logger
from modules.utils import ensure_dir, device_autoselect

log = get_logger(__name__)

def calibrate_threshold(
    success_val_pt: Path,
    vocab_path: Path,
    ckpt_path: Path,
    section_buckets: int = 2048,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    dim_ff: int = 2048,
    dropout: float = 0.1,
    max_len: int = 256,
    batch_size: int = 256,
    percentile: float = 99.5,
    output_path: Path = Path("thresholds.json")
):
    """
    Калибрует порог NLL (Negative Log-Likelihood) для детектирования аномалий.
    
    Процесс:
    1. Загружает обученную модель и валидационные данные (только success-логи)
    2. Вычисляет NLL для каждого примера в валидационном наборе
    3. Находит процентиль (например, 99.5%) от распределения NLL значений
    4. Сохраняет этот порог для использования в детектировании аномалий
    
    Идея: Если NLL нового лога превышает порог, вычисленный на success-логах,
    то это вероятно аномалия.
    """
    # Создаем директорию для выходного файла если нужно
    ensure_dir(output_path.parent)

    # --- Загрузка словаря событий ---
    # Словарь содержит маппинг event_id → индекс токена для модели
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    log.info(f"[Calibrate] Loaded vocab: {vocab_size} events")

    # --- Загрузка валидационных данных ---
    # Загружаем предобработанные тензоры (input_ids, target_ids, sec_ids)
    # Эти данные содержат только success-логи для калибровки порога
    val = load_prepared_tensors(success_val_pt)
    ds = WindowsDataset(val.input_ids, val.target_ids, val.sec_ids)
    
    # Определяем устройство для вычислений (CUDA/MPS/CPU)
    dev = device_autoselect()
    # pin_memory работает только с CUDA (ускоряет передачу данных CPU→GPU)
    use_pin_memory = dev.type == 'cuda'
    # DataLoader для батчевой обработки данных
    # shuffle=False - не перемешиваем, так как нам нужна детерминированность
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    # --- Создание модели с архитектурой из конфига ---
    # Модель должна иметь ту же архитектуру, что и при обучении
    model = UnifiedLogBERT(
        vocab_size=vocab_size,
        section_buckets=section_buckets,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        max_len=max_len,
    )
    
    # --- Загрузка весов обученной модели ---
    # Загружаем чекпоинт модели, обученной на train данных
    # map_location="cpu" - сначала загружаем на CPU, затем переместим на нужное устройство
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    # Переводим модель в режим оценки (отключает dropout, batch norm в eval mode)
    # и перемещаем на выбранное устройство (GPU/CPU)
    model.eval().to(dev)
    
    log.info(f"🖥️  Using device: {dev}")

    # --- Настройка функции потерь ---
    # reduction="none" - возвращает loss для каждого примера отдельно (не усредняет)
    # Это нужно для сбора индивидуальных NLL значений
    ce = nn.CrossEntropyLoss(reduction="none")
    nll_values = []  # Список для накопления всех NLL значений

    # --- Сбор NLL значений на валидационном наборе ---
    # Проходим по всем батчам success-валидационных данных
    # и собираем NLL (Negative Log-Likelihood) для каждого примера
    log.info("[Calibrate] Collecting NLL values on success validation...")
    with torch.no_grad():  # Отключаем вычисление градиентов (не нужно для инференса)
        progress = tqdm(dl, desc="Calibrate [success val]", unit="batch")
        for input_ids, target_ids, sec_ids in progress:
            # Перемещаем данные на выбранное устройство
            input_ids = input_ids.to(dev)
            target_ids = target_ids.to(dev)
            sec_ids = sec_ids.to(dev)
            
            # Forward pass через модель
            # logits: (batch_size, vocab_size) - логиты для предсказания следующего события
            logits = model(input_ids, sec_ids)
            
            # Вычисляем loss для каждого примера в батче
            # loss: (batch_size,) - NLL для каждого примера
            loss = ce(logits, target_ids)
            
            # Сохраняем NLL значения в список (переводим на CPU для экономии памяти GPU)
            nll_values.extend(loss.cpu().tolist())

    # Проверяем, что собрали хотя бы одно значение
    if not nll_values:
        raise RuntimeError("No NLL values collected; check dataset or checkpoint.")

    # --- Вычисление порога по процентилю ---
    # Преобразуем список NLL значений в тензор
    nll_tensor = torch.tensor(nll_values)
    
    # Вычисляем процентиль от распределения NLL значений
    # Например, percentile=99.5 означает, что 99.5% success-логов имеют NLL <= threshold
    # Это означает, что если новый лог имеет NLL > threshold, то он вероятно аномальный
    # Формула: quantile = значение, ниже которого находится percentile% данных
    thr = float(torch.quantile(nll_tensor, percentile/100.0))
    
    # Пример: если percentile=99.5 и thr=3.5, то:
    # - 99.5% success-логов имеют NLL <= 3.5
    # - 0.5% success-логов имеют NLL > 3.5 (это нормально, так как есть вариативность)
    # - Если новый лог имеет NLL > 3.5, он считается аномальным

    # --- Сохранение результата калибровки ---
    # Сохраняем порог в JSON файл для использования в детектировании аномалий
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "percentile": percentile,      # Использованный процентиль (например, 99.5)
            "threshold_nll": thr,          # Вычисленный порог NLL
            "count": len(nll_values)       # Количество примеров, на которых вычислен порог
        }, f, indent=2)

    log.info(f"[Calibrate] threshold_nll={thr:.4f} @p{percentile} ({len(nll_values)} samples)")
    log.info(f"[Calibrate] Saved → {output_path}")

    # --- Визуализация распределения NLL ---
    # Строим гистограмму NLL значений и отображаем выбранный процентиль
    fig_path = output_path.parent / f"nll_distribution_p{percentile:.1f}.png"
    plt.figure(figsize=(8, 5))
    plt.hist(nll_values, bins=200, color="steelblue", alpha=0.7, density=True)
    plt.axvline(thr, color="red", linestyle="--", label=f"p{percentile:.1f}={thr:.2f}")
    plt.title("NLL distribution (success validation)")
    plt.xlabel("Negative log-likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log.info(f"[Calibrate] Saved plot → {fig_path}")

def main():
    """
    Калибровка порога NLL по success-валидации.
    
    Все параметры берутся из config.yaml:
    - val_pt: dataset.dataset/val.pt - валидационные данные (success-логи)
    - vocab: dataset.vocab/event_vocab.json - словарь событий
    - ckpt: model.checkpoints/model.pt - обученная модель
    - percentile: model.percentile - процентиль для порога (по умолчанию 99.5)
    - output: model.checkpoints/thresholds.json - файл с результатом калибровки
    """
    # --- Загрузка конфигурации из config.yaml ---
    # Все пути и параметры берутся из конфигурационного файла
    config = get_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    
    # --- Определение путей к входным файлам из config.yaml ---
    
    # 1. Валидационные данные: dataset.dataset/val.pt
    # Содержит только success-логи для калибровки порога
    dataset_dir = Path(dataset_cfg.get("dataset", "./dataset/train"))
    val_pt = dataset_dir / "val.pt"
    if not val_pt.exists():
        log.error(f"❌ Validation file not found: {val_pt}")
        log.error(f"   Please ensure val.pt exists in {dataset_dir}")
        raise SystemExit(1)
    log.info(f"📁 Using validation file from config: {val_pt}")
    
    # 2. Словарь событий: dataset.vocab/event_vocab.json
    # Содержит маппинг event_id → индекс токена для модели
    vocab_dir = Path(dataset_cfg.get("vocab", "./dataset/vocab"))
    vocab_path = vocab_dir / "event_vocab.json"
    if not vocab_path.exists():
        log.error(f"❌ Vocab file not found: {vocab_path}")
        log.error(f"   Please ensure event_vocab.json exists in {vocab_dir}")
        raise SystemExit(1)
    log.info(f"📁 Using vocab file from config: {vocab_path}")
    
    # 3. Чекпоинт модели: model.checkpoints/model.pt
    # Обученная модель LogBERT для вычисления NLL
    ckpt_dir = Path(model_cfg.get("checkpoints", "./model/checkpoints"))
    ckpt_path = ckpt_dir / "model.pt"
    if not ckpt_path.exists():
        log.error(f"❌ Checkpoint file not found: {ckpt_path}")
        log.error(f"   Please ensure model.pt exists in {ckpt_dir}")
        raise SystemExit(1)
    log.info(f"📁 Using checkpoint from config: {ckpt_path}")
    
    # 4. Процентиль для порога: model.percentile
    # Определяет, какой процент success-логов должен иметь NLL <= threshold
    # Например, 99.5 означает, что 99.5% success-логов имеют NLL <= threshold
    percentile = float(model_cfg.get("percentile", 99.5))
    log.info(f"📊 Using percentile from config: {percentile}")
    
    # 5. Выходной файл: model.checkpoints/thresholds.json
    # Сохраняет вычисленный порог для использования в детектировании аномалий
    output_path = ckpt_dir / "thresholds.json"
    log.info(f"📁 Output file: {output_path}")
    
    # --- Параметры модели из config.yaml ---
    # Эти параметры должны совпадать с параметрами при обучении модели
    section_buckets = int(model_cfg.get("section_buckets", 2048))  # Количество бакетов для секций
    d_model = int(model_cfg.get("d_model", 512))                   # Размерность модели
    n_heads = int(model_cfg.get("n_heads", 8))                      # Количество attention голов
    n_layers = int(model_cfg.get("n_layers", 6))                   # Количество слоев трансформера
    dim_ff = int(model_cfg.get("dim_ff", 2048))                    # Размерность feed-forward слоя
    dropout = float(model_cfg.get("dropout", 0.1))                 # Dropout вероятность
    max_len = int(model_cfg.get("max_len", 512))                   # Максимальная длина последовательности
    batch_size = int(model_cfg.get("batch_size", 256))              # Размер батча для обработки
    
    log.info(f"📊 Model config: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    log.info(f"📊 Training config: batch_size={batch_size}, max_len={max_len}")
    
    calibrate_threshold(
        success_val_pt=val_pt,
        vocab_path=vocab_path,
        ckpt_path=ckpt_path,
        section_buckets=section_buckets,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        max_len=max_len,
        batch_size=batch_size,
        percentile=percentile,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
