# -*- coding: utf-8 -*-
"""
modules/model_utils.py — утилиты для работы с моделями, чекпоинтами и устройствами.

Функции:
  - resolve_checkpoint_paths: разрешает пути к чекпоинтам модели из config
  - get_device_from_config: получает устройство для вычислений из config
  - load_model_from_checkpoint: загружает модель LogBERT из чекпоинта
  - get_vocab_size: получает размер словаря из event_vocab.json

Используется в:
  - 12-train_one_epoch.py (работа с моделью и устройством)
  - 14-run_inference.py (загрузка модели для инференса)
"""
from pathlib import Path
from typing import Tuple, Optional, Union, cast
import json
import torch
import torch.nn as nn

from modules.model import UnifiedLogBERT
from modules.utils import device_autoselect
from modules.logger import get_logger

# Для работы с Config объектом из modules.config
try:
    from modules.config import Config
except ImportError:
    Config = object  # type: ignore

logger = get_logger(__name__)


def resolve_checkpoint_paths(config: Union[Config, dict]) -> Tuple[Path, Path]:
    """
    Разрешает пути к чекпоинтам модели из конфигурации.
    
    Поддерживает два варианта в config.yaml:
    
    1) model.checkpoints: "model/checkpoints/model.pt"
       → model_ckpt = "model/checkpoints/model.pt"
       → thresholds = "model/checkpoints/thresholds.json"
    
    2) model.checkpoints:
         model: "model/checkpoints/model.pt"
         thresholds: "model/checkpoints/thresholds.json"
    
    Args:
        config: Конфигурация (Config объект или dict)
    
    Returns:
        Кортеж (model_ckpt_path, thresholds_path)
    
    Raises:
        ValueError: Если формат model.checkpoints не поддерживается
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(config, 'get'):
        # Config объект
        cp_cfg = config.get("model.checkpoints")
    else:
        # dict
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        cp_cfg = model_cfg.get("checkpoints")
    
    if isinstance(cp_cfg, str):
        p = Path(cp_cfg)
        if p.suffix == ".pt":
            model_ckpt = p
            thresholds = p.with_name("thresholds.json")
        else:
            # считаем, что это директория
            model_ckpt = p / "model.pt"
            thresholds = p / "thresholds.json"
    elif isinstance(cp_cfg, dict):
        model_ckpt = Path(cp_cfg["model"])
        thresholds = Path(cp_cfg["thresholds"])
    else:
        raise ValueError(
            "model.checkpoints в config.yaml должен быть либо строкой, "
            "либо словарём с ключами 'model' и 'thresholds'"
        )

    return model_ckpt, thresholds


def get_device_from_config(config: Union[Config, dict]) -> torch.device:
    """
    Определяет устройство для вычислений на основе config.yaml.
    
    Использует gpu.device из config, если указано и доступно.
    Иначе использует device_autoselect() для автоматического выбора.
    
    Args:
        config: Конфигурация (Config объект или dict)
    
    Returns:
        torch.device: Выбранное устройство (cuda, mps или cpu)
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(config, 'get_gpu_config'):
        # Config объект
        gpu_cfg = config.get_gpu_config()
    else:
        # dict
        gpu_cfg = config.get("gpu", {}) if isinstance(config, dict) else {}
    
    device_name = gpu_cfg.get("device", "").lower() if gpu_cfg else ""
    
    # Если в config указано устройство, проверяем его доступность
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_name == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        # Используем автоматический выбор устройства
        dev = device_autoselect()
        return cast(torch.device, dev)


def load_model_from_checkpoint(
    config: Union[Config, dict],
    checkpoint_path: Path,
    vocab_size: int,
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
) -> nn.Module:
    """
    Загружает модель LogBERT из чекпоинта.
    
    Использует настройки модели из config.yaml для создания архитектуры.
    Загружает веса из checkpoint_path.
    
    Args:
        config: Конфигурация (Config объект или dict)
        checkpoint_path: Путь к файлу чекпоинта модели (.pt)
        vocab_size: Размер словаря (количество токенов)
        device: Устройство для размещения модели (если None, определяется из config)
        eval_mode: Если True, переводит модель в режим оценки (по умолчанию True)
    
    Returns:
        Загруженная модель UnifiedLogBERT
    
    Raises:
        ValueError: Если обязательные параметры модели не указаны в config.yaml
        FileNotFoundError: Если чекпоинт не найден
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(config, 'get'):
        # Config объект
        model_cfg = config.get("model")
    else:
        # dict
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    
    if not model_cfg:
        raise ValueError(
            "Секция 'model' не найдена в config.yaml. "
            "Пожалуйста, добавьте секцию model с параметрами архитектуры."
        )
    
    # Проверяем наличие обязательных параметров
    required_params = ["section_buckets", "d_model", "n_heads", "n_layers", "dim_ff", "dropout", "max_len"]
    missing_params = [p for p in required_params if p not in model_cfg]
    
    if missing_params:
        raise ValueError(
            f"В секции 'model' отсутствуют обязательные параметры: {', '.join(missing_params)}. "
            f"Пожалуйста, укажите все параметры архитектуры модели в config.yaml."
        )
    
    # Создаем модель с параметрами из конфигурации
    model = UnifiedLogBERT(
        vocab_size=vocab_size,
        section_buckets=int(model_cfg["section_buckets"]),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        dim_ff=int(model_cfg["dim_ff"]),
        dropout=float(model_cfg["dropout"]),
        max_len=int(model_cfg["max_len"]),
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Не найден чекпоинт модели: {checkpoint_path}")

    # Загружаем веса модели
    state = torch.load(checkpoint_path, map_location="cpu")
    # Если при тренировке было сохранено с ключом 'model_state_dict'
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    logger.info("✅ Модель загружена из %s", checkpoint_path)

    # Определяем устройство если не указано
    if device is None:
        device = get_device_from_config(config)
    
    model.to(device)
    
    if eval_mode:
        model.eval()
    
    logger.info("🖥️  Используется устройство: %s", device)
    return model


def get_vocab_size(vocab_path: Path) -> int:
    """
    Получает размер словаря из файла event_vocab.json.
    
    Args:
        vocab_path: Путь к файлу словаря (event_vocab.json)
    
    Returns:
        Размер словаря (количество токенов)
    
    Raises:
        FileNotFoundError: Если файл словаря не найден
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Файл словаря не найден: {vocab_path}")
    
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    logger.info("📚 Размер словаря: %d токенов", vocab_size)
    return vocab_size


def get_checkpoint_dir_from_config(config: Union[Config, dict]) -> Path:
    """
    Получает директорию для чекпоинтов из конфигурации.
    
    Извлекает путь из model.checkpoints. Если указан файл, возвращает его директорию.
    Если указана директория, возвращает её.
    
    Args:
        config: Конфигурация (Config объект или dict)
    
    Returns:
        Path к директории с чекпоинтами
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(config, 'get'):
        # Config объект
        cp_cfg = config.get("model.checkpoints")
    else:
        # dict
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        cp_cfg = model_cfg.get("checkpoints")
    
    if isinstance(cp_cfg, str):
        p = Path(cp_cfg)
        if p.suffix == ".pt":
            # Если указан файл, возвращаем его директорию
            return p.parent
        else:
            # Если указана директория, возвращаем её
            return p
    elif isinstance(cp_cfg, dict):
        # Если указан словарь, берем директорию из пути к модели
        model_path = Path(cp_cfg.get("model", "./checkpoints/model.pt"))
        return model_path.parent
    else:
        # По умолчанию
        return Path("./checkpoints")

