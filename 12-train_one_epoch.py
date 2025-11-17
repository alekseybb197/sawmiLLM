#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_one_epoch.py — выполняет одну эпоху обучения LogBERT модели,
с логированием метрик и артефактов в MLflow и системой рекомендаций.

ИСПОЛЬЗОВАНИЕ:
==============

1. Базовый запуск (без параметров):
   ./train_one_epoch.py
   - Автоматически определяет номер эпохи из training_state.json (или начинает с 1)
   - Все пути и параметры берутся из config.yaml

2. Запуск с указанием номера эпохи:
   ./train_one_epoch.py --epoch 5
   - Обучает эпоху 5
   - Все остальные параметры из config.yaml

3. Переопределение путей к данным:
   ./train_one_epoch.py --epoch 1 --train-pt dataset/train.pt --vocab dataset/vocab.json
   ./train_one_epoch.py --epoch 1 --val-pt dataset/val.pt
   ./train_one_epoch.py --epoch 1 --ckpt-dir ./checkpoints --outdir ./reports

4. Переопределение параметров модели:
   ./train_one_epoch.py --epoch 1 --d-model 768 --n-heads 12 --batch-size 128
   ./train_one_epoch.py --epoch 1 --lr 5e-5 --weight-decay 0.01 --grad-clip 1.0

5. Включение Mixed Precision Training (AMP):
   ./train_one_epoch.py --epoch 1 --amp

6. Комбинированный запуск:
   ./train_one_epoch.py --epoch 3 --batch-size 512 --lr 1e-5 --amp --ckpt-dir ./custom_ckpt

ПРИОРИТЕТ ПАРАМЕТРОВ:
====================
1. Аргументы командной строки (высший приоритет)
2. config.yaml (секция model)
3. Значения по умолчанию в TrainConfig

ПУТИ ИЗ CONFIG.YAML:
====================
- train.pt: dataset.dataset/train.pt
- vocab: dataset.vocab/event_vocab.json
- val.pt: dataset.dataset/val.pt (опционально, если файл существует)
- ckpt_dir: model.checkpoints
- outdir: model.reports (обязательно, без исключений)

ОБРАБОТКА ПРЕРЫВАНИЯ (Ctrl-C):
===============================
При нажатии Ctrl-C скрипт корректно сохраняет:
- Состояние модели (model.pt и model_epoch_{N}_interrupted.pt)
- Состояние оптимизатора (optimizer.pt)
- Состояние scaler (scaler.pt, если используется AMP)
- Отчет о прерванном обучении (report_epoch_{N}_interrupted.json)

Для продолжения обучения запустите:
./train_one_epoch.py --epoch {N}

РЕКОМЕНДАЦИИ ПО ОБУЧЕНИЮ:
=========================
Скрипт автоматически оценивает эффективность обучения и выдает рекомендации:
- "continue" - продолжить обучение (улучшение метрик или первая эпоха)
- "stop" - остановить обучение (переобучение или нет улучшения)

Рекомендации основаны на:
- Сравнении train_loss и val_loss
- Отношении val/train (обнаружение переобучения)
- Динамике улучшения validation loss
"""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Union
import json
import os
import signal
import sys
import time
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader
import typer
from tqdm import tqdm

from modules.model import UnifiedLogBERT
from modules.datasets import WindowsDataset, load_prepared_tensors
from modules.utils import ensure_dir, device_autoselect, recommend_action
from modules.logger import get_logger
from modules.config import get_config, Config

log = get_logger(__name__)

# Проверяем наличие psutil для мониторинга ресурсов
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    log.warning("psutil не установлен. Мониторинг ресурсов будет отключен. Установите: pip install psutil")

# Создаем приложение Typer
app = typer.Typer(
    name="train_one_epoch",
    help="Обучение LogBERT для одной эпохи с MLflow tracking и рекомендациями",
    add_completion=False,
)


@dataclass
class TrainConfig:
    """Конфигурация для обучения LogBERT модели."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dim_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 512
    section_buckets: int = 2048
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = False
    seed: int = 42


def load_train_config_from_yaml(config: Config, **overrides) -> TrainConfig:
    """
    Загружает TrainConfig из config.yaml с возможностью переопределения через overrides.
    
    Порядок приоритета: overrides > config.yaml > значения по умолчанию в TrainConfig.
    """
    model_cfg = config.get("model", {})
    
    # Загружаем параметры: сначала из overrides (CLI аргументы), затем из config.yaml, затем дефолты
    return TrainConfig(
        d_model=overrides.get("d_model", model_cfg.get("d_model", 512)),
        n_heads=overrides.get("n_heads", model_cfg.get("n_heads", 8)),
        n_layers=overrides.get("n_layers", model_cfg.get("n_layers", 6)),
        dim_ff=overrides.get("dim_ff", model_cfg.get("dim_ff", 2048)),
        dropout=overrides.get("dropout", model_cfg.get("dropout", 0.1)),
        max_len=overrides.get("max_len", model_cfg.get("max_len", 512)),
        section_buckets=overrides.get("section_buckets", model_cfg.get("section_buckets", 2048)),
        batch_size=overrides.get("batch_size", model_cfg.get("batch_size", 256)),
        lr=overrides.get("lr", model_cfg.get("lr", 1e-4)),
        weight_decay=overrides.get("weight_decay", model_cfg.get("weight_decay", 0.01)),
        grad_clip=overrides.get("grad_clip", model_cfg.get("grad_clip", 1.0)),
        amp=overrides.get("amp", model_cfg.get("amp", False)),
        seed=overrides.get("seed", model_cfg.get("seed", 42)),
    )


def get_resource_stats(device: torch.device) -> dict[str, Any]:
    """
    Собирает статистику использования ресурсов: RAM, CPU, GPU.
    
    Args:
        device: Устройство PyTorch (cuda, mps, cpu)
    
    Returns:
        Словарь со статистикой ресурсов
    """
    stats: dict[str, Any] = {}
    
    if not PSUTIL_AVAILABLE:
        stats['error'] = 'psutil не установлен'
        return stats
    
    # RAM статистика
    mem = psutil.virtual_memory()
    stats['ram'] = {
        'total_gb': round(mem.total / (1024**3), 2),
        'used_gb': round(mem.used / (1024**3), 2),
        'available_gb': round(mem.available / (1024**3), 2),
        'percent': mem.percent
    }
    
    # CPU статистика
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    stats['cpu'] = {
        'usage_percent': cpu_percent,
        'cores': cpu_count
    }
    
    # GPU статистика (для CUDA)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()  # Синхронизация для точных измерений
        gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        stats['gpu'] = {
            'device': torch.cuda.get_device_name(device),
            'memory_allocated_gb': round(gpu_mem_allocated, 2),
            'memory_reserved_gb': round(gpu_mem_reserved, 2),
            'memory_total_gb': round(gpu_mem_total, 2),
            'memory_percent': round((gpu_mem_reserved / gpu_mem_total) * 100, 1) if gpu_mem_total > 0 else 0
        }
    elif device.type == 'mps':
        # Для MPS показываем только базовую информацию
        stats['gpu'] = {
            'device': 'Apple Silicon (MPS)',
            'memory_info': 'N/A (MPS не предоставляет детальную статистику)'
        }
    else:
        stats['gpu'] = {
            'device': 'CPU',
            'memory_info': 'N/A'
        }
    
    return stats


def format_resource_stats(stats: dict[str, Any]) -> str:
    """
    Форматирует статистику ресурсов в читаемый формат для вывода в лог.
    
    Args:
        stats: Словарь со статистикой ресурсов
    
    Returns:
        Отформатированная строка для вывода
    """
    if 'error' in stats:
        return f"⚠️  {stats['error']}"
    
    lines = []
    lines.append("📊 Ресурсы системы:")
    
    # RAM
    ram = stats.get('ram', {})
    lines.append(f"  💾 RAM: {ram.get('used_gb', 0):.1f} GB / {ram.get('total_gb', 0):.1f} GB ({ram.get('percent', 0):.1f}%)")
    
    # CPU
    cpu = stats.get('cpu', {})
    lines.append(f"  🖥️  CPU: {cpu.get('usage_percent', 0):.1f}% ({cpu.get('cores', 0)} cores)")
    
    # GPU
    gpu = stats.get('gpu', {})
    if gpu.get('memory_allocated_gb'):
        lines.append(f"  🎮 GPU ({gpu.get('device', 'Unknown')}):")
        lines.append(f"     - Выделено: {gpu.get('memory_allocated_gb', 0):.2f} GB")
        lines.append(f"     - Зарезервировано: {gpu.get('memory_reserved_gb', 0):.2f} GB")
        lines.append(f"     - Всего: {gpu.get('memory_total_gb', 0):.2f} GB ({gpu.get('memory_percent', 0):.1f}%)")
    else:
        lines.append(f"  🎮 GPU: {gpu.get('device', 'Unknown')} - {gpu.get('memory_info', 'N/A')}")
    
    return "\n".join(lines)


def train_one_epoch(
    epoch: int,
    train_pt: Path,
    val_pt: Path,
    vocab_path: Path,
    ckpt_dir: Path,
    outdir: Path,
    experiment_name: str = "LogBERT_training",
    cfg: TrainConfig | None = None,
    config: Config | None = None,
):
    """
    Выполняет одну эпоху обучения LogBERT модели.
    
    Args:
        epoch: Номер эпохи
        train_pt: Путь к файлу с обучающими данными (.pt)
        val_pt: Путь к файлу с валидационными данными (.pt), может быть None
        vocab_path: Путь к файлу словаря событий (JSON)
        ckpt_dir: Директория для сохранения чекпоинтов
        outdir: Директория для сохранения отчетов
        experiment_name: Имя эксперимента в MLflow (переопределяется из config.yaml если указано)
        cfg: Конфигурация обучения (если None, загружается из config.yaml)
        config: Полный объект Config (если None, загружается из get_config())
    """
    # Загружаем конфигурацию если не передана
    if config is None:
        config = get_config()
    if cfg is None:
        cfg = load_train_config_from_yaml(config)
    
    # Создаем директории для выходных файлов и чекпоинтов
    ensure_dir(outdir)
    ensure_dir(ckpt_dir)
    
    # Переменные для хранения состояния модели (для обработки Ctrl-C)
    # Сохраняем ссылку на cfg для обработчика прерывания
    cfg_ref: TrainConfig = cfg
    model_state: Optional[torch.nn.Module] = None
    optim_state: Optional[torch.optim.Optimizer] = None
    scaler_state: Optional[torch.amp.GradScaler] = None
    mlflow_run_context_obj: Optional[object] = None
    mlflow_enabled_flag = False
    training_interrupted = False
    # Переменные для отслеживания прогресса обучения
    train_progress_obj: Optional[object] = None
    training_start_time: float = 0.0
    total_batches_count: int = 0
    current_batch_count: int = 0
    current_batch_loss: Optional[float] = None
    current_avg_loss: Optional[float] = None
    device_ref: Optional[torch.device] = None
    
    def handle_interrupt(signum, frame):
        """Обработчик сигнала прерывания (Ctrl-C) - сохраняет состояние модели и завершает обучение."""
        nonlocal training_interrupted, current_batch_loss, current_avg_loss, device_ref
        training_interrupted = True
        log.warning("\n⚠️  Получен сигнал прерывания (Ctrl-C). Сохраняю состояние модели...")
        
        try:
            # Сохраняем модель если она была инициализирована
            if model_state is not None:
                # Сохраняем модель как финальный чекпоинт эпохи
                interrupted_model_path = ckpt_dir / f"model_epoch_{epoch}_interrupted.pt"
                torch.save(model_state.state_dict(), interrupted_model_path)
                log.info(f"💾 Сохранена прерванная модель: {interrupted_model_path}")
                
                # Сохраняем последнюю модель (перезаписываем model.pt)
                final_model_path = ckpt_dir / "model.pt"
                torch.save(model_state.state_dict(), final_model_path)
                log.info(f"💾 Сохранена финальная модель: {final_model_path}")
                
                # Сохраняем состояние оптимизатора
                if optim_state is not None:
                    torch.save(optim_state.state_dict(), ckpt_dir / "optimizer.pt")
                    log.info(f"💾 Сохранено состояние оптимизатора")
                
                # Сохраняем состояние scaler если используется AMP на CUDA
                # Определяем, был ли AMP включен (проверяем тип устройства через cfg_ref)
                if scaler_state is not None and cfg_ref is not None:
                    # Проверяем, что scaler был активен (не пустой)
                    try:
                        scaler_state_dict = scaler_state.state_dict()
                        if scaler_state_dict:
                            torch.save(scaler_state_dict, ckpt_dir / "scaler.pt")
                            log.info(f"💾 Сохранено состояние scaler")
                    except Exception:
                        pass  # Если scaler не активен, пропускаем
                
                # Сохраняем состояние обучения с меткой прерывания и прогрессом
                # Используем глобальные переменные для получения информации о прогрессе
                elapsed_time = time.time() - training_start_time if training_start_time > 0 else 0
                progress_pct = (current_batch_count / total_batches_count * 100) if total_batches_count > 0 else 0
                
                # Получаем статистику ресурсов для сохранения
                if device_ref is not None:
                    resource_stats = get_resource_stats(device_ref)
                else:
                    resource_stats = {}
                
                progress_info = {
                    "current_batch": current_batch_count,
                    "total_batches": total_batches_count,
                    "progress_pct": round(progress_pct, 1),
                    "elapsed_time_sec": round(elapsed_time, 1),
                    "elapsed_time_min": round(elapsed_time / 60, 2),
                    "current_loss": round(current_batch_loss, 4) if current_batch_loss is not None else None,
                    "avg_loss": round(current_avg_loss, 4) if current_avg_loss is not None else None,
                    "batches_per_sec": round(current_batch_count / elapsed_time, 2) if elapsed_time > 0 else 0,
                    "status": f"interrupted at {current_batch_count}/{total_batches_count} ({progress_pct:.1f}%)"
                }
                
                interrupted_state = {
                    "epoch": epoch,
                    "best": False,
                    "interrupted": True,
                    "progress": progress_info,
                    "resources": resource_stats,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with (ckpt_dir / "training_state.json").open("w", encoding="utf-8") as f:
                    json.dump(interrupted_state, f, indent=2)
                
                # Сохраняем отчет о прерванном обучении
                interrupted_report = {
                    "epoch": epoch,
                    "status": "interrupted",
                    "train_loss": "N/A (training interrupted)",
                    "val_loss": None,
                    "time_min": "N/A",
                    "val_train_ratio": None,
                    "recommendation": {"action": "interrupted", "reason": "Training interrupted by user (Ctrl-C)"},
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                report_path = outdir / f"report_epoch_{epoch}_interrupted.json"
                json.dump(interrupted_report, open(report_path, "w"), indent=2)
                log.info(f"💾 Сохранен отчет о прерванном обучении: {report_path}")
                
                # Логируем статус в MLflow если он активен
                if mlflow_enabled_flag and mlflow_run_context_obj is not None:
                    try:
                        mlflow.set_tag("status", "interrupted")
                        mlflow.set_tag("reason", "Training interrupted by user (Ctrl-C)")
                        mlflow.log_param("interrupted", True)
                        mlflow.log_artifact(str(report_path))
                        log.info(f"📊 MLflow: статус прерывания залогирован")
                    except Exception as e:
                        log.warning(f"⚠️  Не удалось залогировать в MLflow: {e}")
            
            log.info(f"✅ Состояние модели сохранено. Обучение прервано на эпохе {epoch}.")
            log.info(f"   Для продолжения запустите: ./train_one_epoch.py --epoch {epoch}")
            
        except Exception as e:
            log.error(f"❌ Ошибка при сохранении состояния: {e}")
            import traceback
            traceback.print_exc()
        
        # Завершаем программу
        sys.exit(0)
    
    # Устанавливаем обработчик сигнала прерывания
    signal.signal(signal.SIGINT, handle_interrupt)

    # --- MLflow setup ---
    # Читаем настройки MLflow из config.yaml
    model_cfg = config.get("model", {})
    mlflow_cfg = model_cfg.get("mlflow", {})
    mlflow_enabled = mlflow_cfg.get("enabled", True)
    mlflow_enabled_flag = mlflow_enabled  # Сохраняем для обработчика прерывания
    
    # Если MLflow включен, настраиваем подключение
    if mlflow_enabled:
        # URL из config.yaml или переменной окружения или дефолт
        mlflow_url = mlflow_cfg.get("url", os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_tracking_uri(mlflow_url)
        
        # Имя эксперимента из config.yaml или переданное значение
        mlflow_experiment_name = mlflow_cfg.get("experiment_name", experiment_name)
        mlflow.set_experiment(mlflow_experiment_name)
        log.info(f"MLflow enabled: tracking_uri={mlflow_url}, experiment={mlflow_experiment_name}")
    else:
        log.info("MLflow disabled (model.mlflow.enabled: false)")

    # --- Load vocab ---
    # Загружаем словарь событий для определения размера словаря модели
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # --- Load or init model ---
    # Выбираем устройство (CUDA, MPS или CPU) для определения pin_memory
    dev = device_autoselect()
    device_ref = dev  # Сохраняем для использования в handle_interrupt
    log.info(f"🖥️  Используемое устройство: {dev}")
    # pin_memory работает только с CUDA, поэтому проверяем тип устройства
    use_pin_memory = dev.type == 'cuda'
    
    # --- Load data ---
    # Загружаем обучающие данные
    tr = load_prepared_tensors(train_pt)
    train_ds = WindowsDataset(tr.input_ids, tr.target_ids, tr.sec_ids)
    # DataLoader с перемешиванием для обучения
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=use_pin_memory)
    
    # Загружаем валидационные данные (если указаны и файл существует)
    val_dl = None
    if val_pt and val_pt.exists():
        vl = load_prepared_tensors(val_pt)
        val_ds = WindowsDataset(vl.input_ids, vl.target_ids, vl.sec_ids)
        # DataLoader без перемешивания для валидации
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    # Создаем модель LogBERT с параметрами из конфигурации
    model = UnifiedLogBERT(
        vocab_size=vocab_size,
        section_buckets=cfg.section_buckets,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    )
    # Перемещаем модель на выбранное устройство
    model.to(dev)
    
    # --- Load checkpoint (model, optimizer, scaler, epoch) ---
    # Определяем пути к файлам чекпоинтов
    ckpt_path = ckpt_dir / "model.pt"
    optim_ckpt_path = ckpt_dir / "optimizer.pt"
    scaler_ckpt_path = ckpt_dir / "scaler.pt"
    state_path = ckpt_dir / "training_state.json"
    
    # Создаем оптимизатор и scaler для mixed precision training
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Определяем backend для GradScaler: AMP поддерживается только для CUDA
    # MPS имеет собственную поддержку mixed precision, но не через GradScaler
    amp_backend = 'cuda' if dev.type == 'cuda' else 'cpu'
    # Включаем AMP только если явно указано в конфиге И устройство - CUDA
    amp_enabled = cfg.amp and dev.type == 'cuda'
    scaler = torch.amp.GradScaler(amp_backend, enabled=amp_enabled)
    
    # Логируем информацию об устройстве и AMP
    if cfg.amp and dev.type != 'cuda':
        log.info(f"ℹ️  AMP отключен (требуется CUDA, используется {dev.type})")
    elif amp_enabled:
        log.info(f"✅ AMP включен (CUDA backend)")
    
    # Загружаем чекпоинты если они существуют (для продолжения обучения)
    if ckpt_path.exists():
        # Загружаем веса модели
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        log.info(f"Loaded model checkpoint: {ckpt_path.name}")
        
        # Загружаем состояние оптимизатора (если есть)
        if optim_ckpt_path.exists():
            optim.load_state_dict(torch.load(optim_ckpt_path, map_location="cpu"))
            log.info(f"Loaded optimizer checkpoint: {optim_ckpt_path.name}")
        
        # Загружаем состояние scaler (если используется AMP на CUDA и файл существует)
        if scaler_ckpt_path.exists() and amp_enabled:
            scaler.load_state_dict(torch.load(scaler_ckpt_path, map_location="cpu"))
            log.info(f"Loaded scaler checkpoint: {scaler_ckpt_path.name}")
    
    # Функция потерь для классификации токенов
    criterion = nn.CrossEntropyLoss()

    # --- Start MLflow run (если включен) ---
    # Используем контекстный менеджер только если MLflow включен
    # Если MLflow отключен, используем пустой контекстный менеджер
    mlflow_run_context: object
    if mlflow_enabled:
        mlflow_run_context = mlflow.start_run(run_name=f"epoch_{epoch}")
    else:
        # Создаем пустой контекстный менеджер для совместимости
        mlflow_run_context = nullcontext()
    
    # Сохраняем ссылку на контекст MLflow для обработчика прерывания
    mlflow_run_context_obj = mlflow_run_context
    
    with mlflow_run_context as run:
        # run может быть None если MLflow отключен, проверяем тип
        if mlflow_enabled and run is not None and hasattr(run, 'info'):
            run_id = getattr(run.info, 'run_id', None)
        else:
            run_id = None
        
        # Логируем параметры в MLflow только если включен
        if mlflow_enabled:
            log.info(f"MLflow run: {run_id}")
            mlflow.log_params({
                "epoch": epoch,
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers,
                "dim_ff": cfg.dim_ff,
                "dropout": cfg.dropout,
                "max_len": cfg.max_len,
                "section_buckets": cfg.section_buckets,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "grad_clip": cfg.grad_clip,
                "amp": cfg.amp,
                "seed": cfg.seed,
            })

        # Сохраняем ссылки на модель, оптимизатор и scaler для обработчика прерывания
        model_state = model
        optim_state = optim
        scaler_state = scaler
        
        # --- Training ---
        # Переводим модель в режим обучения (включает dropout, batch norm в train mode)
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        training_start_time = t0
        
        # Подсчитываем общее количество батчей для отображения прогресса
        total_batches = len(train_dl)
        total_batches_count = total_batches
        current_batch = 0
        current_batch_count = 0
        
        # Проходим по всем батчам обучающих данных с прогрессбаром
        train_progress = tqdm(train_dl, desc=f"Epoch {epoch} [Train]", unit="batch", total=total_batches)
        train_progress_obj = train_progress  # Сохраняем для обработчика прерывания
        for input_ids, target_ids, sec_ids in train_progress:
            # Проверяем флаг прерывания перед каждой итерацией
            if training_interrupted:
                break
            # Перемещаем данные на выбранное устройство (GPU или CPU)
            input_ids, target_ids, sec_ids = [x.to(dev) for x in (input_ids, target_ids, sec_ids)]
            
            # Обнуляем градиенты перед новым forward pass
            optim.zero_grad(set_to_none=True)
            
            # Forward pass с mixed precision training (если включено)
            # Используем правильный backend для autocast в зависимости от устройства
            autocast_context: Union[object, Any]
            if dev.type == 'cuda':
                ##autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) deprecated
                autocast_context = torch.amp.autocast('cuda', enabled=True)
            else:
                # MPS и CPU не используют autocast (MPS имеет встроенную поддержку FP16)
                autocast_context = nullcontext()
            
            with autocast_context:
                logits = model(input_ids, sec_ids)
                loss = criterion(logits, target_ids)
            
            # Backward pass с масштабированием градиентов (для mixed precision)
            # Используем scaler только для CUDA с AMP, иначе обычный backward
            if amp_enabled:
                scaler.scale(loss).backward()
                
                # Обрезаем градиенты если указан порог (предотвращает взрыв градиентов)
                if cfg.grad_clip:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                
                # Обновляем веса модели
                scaler.step(optim)
                scaler.update()
            else:
                # Обычный backward для MPS и CPU (без scaler)
                loss.backward()
                
                # Обрезаем градиенты если указан порог
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                
                # Обновляем веса модели
                optim.step()
            
            # Накопление среднего loss (с учетом размера батча)
            batch_loss = loss.item()
            total_loss += batch_loss * input_ids.size(0)
            n += input_ids.size(0)
            current_batch += 1
            current_batch_count = current_batch
            
            # Сохраняем текущие значения loss для использования в handle_interrupt
            current_batch_loss = batch_loss
            current_avg_loss = total_loss / max(1, n)
            
            # Обновляем прогрессбар с текущим loss
            train_progress.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # Периодический вывод подробного статуса каждые 300 батчей
            if current_batch % 300 == 0:
                elapsed_time = time.time() - t0
                avg_loss = total_loss / max(1, n)
                progress_pct = (current_batch / total_batches * 100) if total_batches > 0 else 0
                eta_seconds = (elapsed_time / current_batch * (total_batches - current_batch)) if current_batch > 0 else 0
                batches_per_sec = current_batch / elapsed_time if elapsed_time > 0 else 0
                
                # Получаем статистику ресурсов
                resource_stats = get_resource_stats(dev)
                
                # Выводим подробный статус в лог
                log.info("=" * 80)
                log.info(f"📈 Статус обучения (Batch {current_batch}/{total_batches}, {progress_pct:.1f}%)")
                log.info(f"  ⏱️  Время: {elapsed_time/60:.1f} мин | ETA: {eta_seconds/60:.1f} мин")
                log.info(f"  📊 Loss: текущий={batch_loss:.4f}, средний={avg_loss:.4f}")
                log.info(f"  🚀 Скорость: {batches_per_sec:.2f} batches/sec")
                log.info(format_resource_stats(resource_stats))
                log.info("=" * 80)
            
            # Обновляем training_state.json периодически (каждые 10 батчей или при прерывании)
            if current_batch % 10 == 0 or training_interrupted:
                elapsed_time = time.time() - t0
                avg_loss = total_loss / max(1, n)
                progress_pct = (current_batch / total_batches * 100) if total_batches > 0 else 0
                eta_seconds = (elapsed_time / current_batch * (total_batches - current_batch)) if current_batch > 0 else 0
                
                # Получаем статистику ресурсов для сохранения в state
                resource_stats = get_resource_stats(dev)
                
                training_progress_state = {
                    "epoch": epoch,
                    "best": False,
                    "interrupted": training_interrupted,
                    "progress": {
                        "current_batch": current_batch,
                        "total_batches": total_batches,
                        "progress_pct": round(progress_pct, 1),
                        "elapsed_time_sec": round(elapsed_time, 1),
                        "elapsed_time_min": round(elapsed_time / 60, 2),
                        "eta_seconds": round(eta_seconds, 1),
                        "eta_min": round(eta_seconds / 60, 2),
                        "current_loss": round(batch_loss, 4),
                        "avg_loss": round(avg_loss, 4),
                        "batches_per_sec": round(current_batch / elapsed_time, 2) if elapsed_time > 0 else 0,
                        "status": f"{current_batch}/{total_batches} ({progress_pct:.1f}%)"
                    },
                    "resources": resource_stats,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with (ckpt_dir / "training_state.json").open("w", encoding="utf-8") as f:
                    json.dump(training_progress_state, f, indent=2)
        
        # Вычисляем средний loss за эпоху (если обучение не было прервано)
        if training_interrupted:
            log.warning("⚠️  Обучение было прервано во время тренировочного цикла")
            # Если обучение прервано, используем последнее значение loss
            train_loss = total_loss / max(1, n) if n > 0 else 0.0
        else:
            train_loss = total_loss / max(1, n)

        # --- Validation ---
        # Вычисляем validation loss (если есть валидационные данные и обучение не прервано)
        val_loss = None
        if val_dl is not None and not training_interrupted:
            # Переводим модель в режим оценки (отключает dropout, batch norm в eval mode)
            model.eval()
            total_val, nv = 0.0, 0
            
            # Отключаем вычисление градиентов для валидации (ускоряет процесс)
            with torch.no_grad():
                val_progress = tqdm(val_dl, desc=f"Epoch {epoch} [Val]", unit="batch")
                for input_ids, target_ids, sec_ids in val_progress:
                    # Перемещаем данные на устройство
                    input_ids, target_ids, sec_ids = [x.to(dev) for x in (input_ids, target_ids, sec_ids)]
                    
                    # Forward pass без backpropagation
                    logits = model(input_ids, sec_ids)
                    loss = criterion(logits, target_ids)
                    
                    # Накопление среднего validation loss
                    batch_val_loss = loss.item()
                    total_val += batch_val_loss * input_ids.size(0)
                    nv += input_ids.size(0)
                    
                    # Обновляем прогрессбар с текущим validation loss
                    val_progress.set_postfix({"val_loss": f"{batch_val_loss:.4f}"})
            
            # Вычисляем средний validation loss за эпоху
            val_loss = total_val / max(1, nv)
        
        # Вычисляем время выполнения эпохи
        dt = time.time() - t0

        # --- Metrics and report ---
        # Вычисляем отношение val/train loss для обнаружения переобучения
        ratio = val_loss / max(train_loss, 1e-9) if val_loss is not None else None
        
        # Загружаем отчет предыдущей эпохи (если есть) для сравнения метрик
        prev_report = None
        if epoch > 1:
            prev_path = outdir / f"report_epoch_{epoch-1}.json"
            if prev_path.exists():
                prev_report = json.load(open(prev_path))
        
        # Получаем рекомендацию о продолжении/остановке обучения
        # Если есть validation loss - используем его, иначе используем train_loss
        if val_loss is not None:
            rec = recommend_action(train_loss, val_loss, prev_report)
        else:
            # Если нет валидации - всегда продолжаем
            rec = {"action": "continue", "reason": "no validation"}

        # --- Save artifacts (model, optimizer, scaler) ---
        # Сохраняем модель для текущей эпохи (с номером эпохи в имени)
        if training_interrupted:
            # Если обучение прервано, сохраняем с меткой interrupted
            model_path = ckpt_dir / f"model_epoch_{epoch}_interrupted.pt"
        else:
            model_path = ckpt_dir / f"model_epoch_{epoch}.pt"
        
        torch.save(model.state_dict(), model_path)
        
        # Сохраняем последнюю модель (перезаписываем model.pt)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        
        # Сохраняем состояние оптимизатора (для продолжения обучения)
        torch.save(optim.state_dict(), ckpt_dir / "optimizer.pt")
        
        # Сохраняем состояние scaler только если используется AMP на CUDA
        if amp_enabled:
            torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")
        
        # Сохраняем состояние обучения (номер эпохи, флаг лучшей модели, статус прерывания, прогресс)
        elapsed_time = time.time() - t0
        progress_pct = (current_batch / total_batches * 100) if total_batches > 0 else 100
        avg_loss = total_loss / max(1, n)
        final_batch_loss = batch_loss if 'batch_loss' in locals() else None
        
        training_state = {
            "epoch": epoch,
            "best": False,
            "interrupted": training_interrupted,
            "progress": {
                "current_batch": current_batch,
                "total_batches": total_batches,
                "progress_pct": round(progress_pct, 1),
                "elapsed_time_sec": round(elapsed_time, 1),
                "elapsed_time_min": round(elapsed_time / 60, 2),
                "current_loss": round(final_batch_loss, 4) if final_batch_loss is not None else None,
                "avg_loss": round(avg_loss, 4),
                "batches_per_sec": round(current_batch / elapsed_time, 2) if elapsed_time > 0 else 0,
                "status": f"{current_batch}/{total_batches} ({progress_pct:.1f}%)" + (" - interrupted" if training_interrupted else " - completed")
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with (ckpt_dir / "training_state.json").open("w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=2)
        
        # Формируем отчет о текущей эпохе
        if training_interrupted:
            report = {
                "epoch": epoch,
                "status": "interrupted",
                "train_loss": train_loss if not training_interrupted else "N/A (training interrupted)",
                "val_loss": val_loss,
                "time_min": round(dt / 60, 2) if dt > 0 else "N/A",
                "val_train_ratio": ratio,
                "recommendation": {"action": "interrupted", "reason": "Training interrupted by user (Ctrl-C)"},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            # Сохраняем отчет с меткой interrupted
            report_path = outdir / f"report_epoch_{epoch}_interrupted.json"
        else:
            report = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_min": round(dt / 60, 2),
                "val_train_ratio": ratio,
                "recommendation": rec,
            }
            # Сохраняем отчет в JSON файл
            report_path = outdir / f"report_epoch_{epoch}.json"
        
        json.dump(report, open(report_path,"w"), indent=2)

        # --- Log to MLflow (если включен) ---
        if mlflow_enabled:
            if training_interrupted:
                # Если обучение прервано, логируем только статус прерывания
                mlflow.set_tag("status", "interrupted")
                mlflow.set_tag("reason", "Training interrupted by user (Ctrl-C)")
                mlflow.log_param("interrupted", True)
                mlflow.log_artifact(str(report_path))
                mlflow.log_artifact(str(model_path))
            else:
                # Подготавливаем метрики для логирования
                metrics = {"train_loss": train_loss, "time_min": dt / 60}
                
                # Добавляем validation метрики если они есть
                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                    metrics["val_train_ratio"] = ratio or 0.0
                
                # Логируем метрики в MLflow
                mlflow.log_metrics(metrics)
                
                # Логируем артефакты (отчет и модель)
                mlflow.log_artifact(str(report_path))
                mlflow.log_artifact(str(model_path))
                
                # Устанавливаем теги для фильтрации в MLflow UI
                mlflow.set_tag("status", rec["action"])
                mlflow.set_tag("reason", rec["reason"])
                mlflow.log_param("hostname", os.uname().nodename)
        
        # Логируем результаты в консоль независимо от MLflow
        if training_interrupted:
            log.warning(f"⚠️  Обучение прервано на эпохе {epoch}")
            log.info(f"[Epoch {epoch}] train_loss={train_loss:.4f} (частично), time={dt/60:.1f}m")
            log.info(f"💾 Состояние модели сохранено. Для продолжения запустите: ./train_one_epoch.py --epoch {epoch}")
        else:
            if val_loss is not None:
                log.info(f"[Epoch {epoch}] train={train_loss:.4f}, val={val_loss:.4f}, ratio={ratio:.3f}, time={dt/60:.1f}m")
            else:
                log.info(f"[Epoch {epoch}] train={train_loss:.4f}, time={dt/60:.1f}m")
            log.info(f"Recommendation: {rec['action']} ({rec['reason']})")
        
        if mlflow_enabled:
            log.info(f"MLflow run complete → {mlflow.get_tracking_uri()} (run_id={run_id})")

@app.command()
def main(
    epoch: Optional[int] = typer.Option(None, "--epoch", "-e", help="Номер эпохи обучения (по умолчанию: определяется из training_state.json или 1)"),
    train_pt: Optional[Path] = typer.Option(None, "--train-pt", help="Путь к файлу с обучающими данными (.pt)"),
    vocab: Optional[Path] = typer.Option(None, "--vocab", "-v", help="Путь к файлу словаря событий (JSON)"),
    ckpt_dir: Optional[Path] = typer.Option(None, "--ckpt-dir", help="Директория для сохранения чекпоинтов"),
    outdir: Optional[Path] = typer.Option(None, "--outdir", "-o", help="Директория для сохранения отчетов"),
    val_pt: Optional[Path] = typer.Option(None, "--val-pt", help="Путь к файлу с валидационными данными (.pt, опционально)"),
    experiment_name: str = typer.Option("LogBERT_training", "--experiment-name", help="Имя эксперимента в MLflow"),
    # Параметры модели (могут переопределить значения из config.yaml)
    d_model: Optional[int] = typer.Option(None, "--d-model", help="Размерность модели (переопределяет config.yaml)"),
    n_heads: Optional[int] = typer.Option(None, "--n-heads", help="Количество attention голов (переопределяет config.yaml)"),
    n_layers: Optional[int] = typer.Option(None, "--n-layers", help="Количество слоев трансформера (переопределяет config.yaml)"),
    dim_ff: Optional[int] = typer.Option(None, "--dim-ff", help="Размерность feed-forward слоя (переопределяет config.yaml)"),
    dropout: Optional[float] = typer.Option(None, "--dropout", help="Вероятность dropout (переопределяет config.yaml)"),
    max_len: Optional[int] = typer.Option(None, "--max-len", help="Максимальная длина последовательности (переопределяет config.yaml)"),
    section_buckets: Optional[int] = typer.Option(None, "--section-buckets", help="Количество секций для bucket encoding (переопределяет config.yaml)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Размер батча (переопределяет config.yaml)"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate (переопределяет config.yaml)"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay", help="Weight decay для оптимизатора (переопределяет config.yaml)"),
    grad_clip: Optional[float] = typer.Option(None, "--grad-clip", help="Порог обрезки градиентов (переопределяет config.yaml)"),
    amp: bool = typer.Option(False, "--amp", help="Использовать mixed precision training (AMP)"),
) -> None:
    """
    Выполняет одну эпоху обучения LogBERT модели с логированием в MLflow.
    
    Все параметры модели сначала загружаются из config.yaml (секция model),
    затем переопределяются аргументами командной строки (если указаны).
    
    Пути к файлам (train_pt, vocab, ckpt_dir, outdir) берутся из config.yaml,
    если не указаны явно через аргументы командной строки.
    
    Путь для отчетов (outdir) берется из model.reports в config.yaml.
    
    Примеры использования:
        # Базовый запуск с параметрами из config.yaml (требуется только --epoch)
        ./train_one_epoch.py --epoch 1
        
        # С переопределением путей
        ./train_one_epoch.py --epoch 1 --train-pt dataset/train.pt --vocab dataset/vocab.json
        
        # С переопределением параметров модели
        ./train_one_epoch.py --epoch 1 --d-model 768 --batch-size 128
    """
    # Загружаем конфигурацию из config.yaml
    config = get_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    
    # Определяем номер эпохи: из аргумента, из training_state.json, или 1 по умолчанию
    if epoch is None:
        # Пытаемся определить номер эпохи из training_state.json
        ckpt_dir_default = Path(model_cfg.get("checkpoints", "./checkpoints"))
        state_path = ckpt_dir_default / "training_state.json"
        
        if state_path.exists():
            try:
                with state_path.open("r", encoding="utf-8") as f:
                    state = json.load(f)
                    epoch_val = state.get("epoch", 0)
                    # Приводим к int, так как JSON может возвращать float
                    epoch = int(epoch_val) + 1 if isinstance(epoch_val, (int, float)) else 1
                    log.info(f"📊 Определена эпоха из training_state.json: {epoch}")
            except Exception as e:
                log.warning(f"⚠️  Не удалось прочитать training_state.json: {e}")
                epoch = 1
        else:
            epoch = 1
            log.info(f"📊 Начинаем обучение с эпохи {epoch} (training_state.json не найден)")
    else:
        log.info(f"📊 Используется указанная эпоха: {epoch}")
    
    # Определяем пути из config.yaml, если не указаны явно
    # train_pt: dataset.dataset/train.pt
    if train_pt is None:
        dataset_dir = Path(dataset_cfg.get("dataset", "./dataset/train"))
        train_pt = dataset_dir / "train.pt"
        if not train_pt.exists():
            log.error(f"❌ Train file not found: {train_pt}")
            log.error(f"   Please specify --train-pt or ensure train.pt exists in {dataset_dir}")
            raise typer.Exit(code=1)
        log.info(f"📁 Using train file from config: {train_pt}")
    
    # vocab: dataset.vocab/event_vocab.json
    if vocab is None:
        vocab_dir = Path(dataset_cfg.get("vocab", "./dataset/vocab"))
        vocab = vocab_dir / "event_vocab.json"
        if not vocab.exists():
            log.error(f"❌ Vocab file not found: {vocab}")
            log.error(f"   Please specify --vocab or ensure event_vocab.json exists in {vocab_dir}")
            raise typer.Exit(code=1)
        log.info(f"📁 Using vocab file from config: {vocab}")
    
    # ckpt_dir: model.checkpoints
    if ckpt_dir is None:
        ckpt_dir_path = model_cfg.get("checkpoints", "./checkpoints")
        ckpt_dir = Path(ckpt_dir_path)
        log.info(f"📁 Using checkpoints directory from config: {ckpt_dir}")
    
    # outdir: всегда используем model.reports из config.yaml (обязательно)
    if outdir is None:
        reports_path = model_cfg.get("reports")
        if not reports_path:
            log.error("❌ model.reports не указан в config.yaml")
            log.error("   Пожалуйста, укажите model.reports в секции model конфигурационного файла")
            raise typer.Exit(code=1)
        outdir = Path(reports_path)
        log.info(f"📁 Using output directory from config (model.reports): {outdir}")
    
    # val_pt: dataset.dataset/val.pt (опционально, если файл существует)
    if val_pt is None:
        dataset_dir = Path(dataset_cfg.get("dataset", "./dataset/train"))
        val_pt_candidate = dataset_dir / "val.pt"
        if val_pt_candidate.exists():
            val_pt = val_pt_candidate
            log.info(f"📁 Using validation file from config: {val_pt}")
        else:
            log.info(f"ℹ️  Validation file not found: {val_pt_candidate} (will skip validation)")
    
    # Формируем словарь переопределений из аргументов командной строки
    # (только те параметры, которые были явно указаны)
    overrides: dict[str, Any] = {}
    if d_model is not None:
        overrides["d_model"] = int(d_model)
    if n_heads is not None:
        overrides["n_heads"] = int(n_heads)
    if n_layers is not None:
        overrides["n_layers"] = int(n_layers)
    if dim_ff is not None:
        overrides["dim_ff"] = int(dim_ff)
    if dropout is not None:
        overrides["dropout"] = float(dropout)
    if max_len is not None:
        overrides["max_len"] = int(max_len)
    if section_buckets is not None:
        overrides["section_buckets"] = int(section_buckets)
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    if lr is not None:
        overrides["lr"] = float(lr)
    if weight_decay is not None:
        overrides["weight_decay"] = float(weight_decay)
    if grad_clip is not None:
        overrides["grad_clip"] = float(grad_clip)
    # AMP: передаем в overrides только если явно указан через --amp
    # Если --amp не указан, используем значение из config.yaml
    # Проверяем через sys.argv, был ли указан флаг --amp
    if "--amp" in sys.argv:
        overrides["amp"] = True
    # Если --amp не указан, не добавляем в overrides - будет использовано значение из config.yaml
    
    # Загружаем конфигурацию из config.yaml, переопределяем аргументами командной строки
    cfg = load_train_config_from_yaml(config, **overrides)
    
    # Логируем загруженную конфигурацию
    log.info(f"Loaded train config from config.yaml (model section)")
    log.info(f"  Model: d_model={cfg.d_model}, n_heads={cfg.n_heads}, n_layers={cfg.n_layers}")
    log.info(f"  Training: batch_size={cfg.batch_size}, lr={cfg.lr}, amp={cfg.amp}")
    
    # Вызываем функцию обучения одной эпохи
    train_one_epoch(
        epoch=epoch,
        train_pt=train_pt,
        val_pt=val_pt or Path(""),  # Путь будет проверен внутри функции
        vocab_path=vocab,
        ckpt_dir=ckpt_dir,
        outdir=outdir,
        experiment_name=experiment_name,
        cfg=cfg,
        config=config  # Передаем полный config для доступа к MLflow настройкам
    )


if __name__ == "__main__":
    app()
