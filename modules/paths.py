# -*- coding: utf-8 -*-
"""
modules/paths.py — утилиты для работы с путями и парсингом имен файлов.

Функции:
  - parse_filename: парсит имя файла формата pipeline_id-build_id-*.lst
  - resolve_dataset_paths: извлекает пути из конфигурации dataset
  - get_vocab_path_from_config: получает путь к словарю из config
  - get_dataset_path_from_config: получает путь из dataset секции
  - get_window_size_from_config: получает размер окна из config
  - get_inputs_from_drain_list: получает входные файлы из dataset.drain_list

Используется в:
  - 09-drain_dataset.py (парсинг имен файлов)
  - 10-build_vocab.py (извлечение путей из конфига)
  - 11-prepare_windows.py (извлечение путей и настроек из конфига)
  - 14-run_inference.py (парсинг имен файлов и извлечение путей)
"""
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, cast, Union

# Для работы с Config объектом из modules.config
try:
    from modules.config import Config
except ImportError:
    Config = Any  # type: ignore


def parse_filename(path: Path) -> Tuple[str, str]:
    """
    Парсит имя файла формата pipeline_id-build_id-*.lst и извлекает pipeline_id и build_id.
    
    Ожидаемый формат: pipeline_id-build_id-logs_content.lst
    Примеры:
      - "207920-26884038-logs_content.lst" -> ("207920", "26884038")
      - "123-456-something.lst" -> ("123", "456")
    
    Args:
        path: Путь к файлу (Path объект)
    
    Returns:
        Кортеж (pipeline_id, build_id)
        Если формат не соответствует ожидаемому, возвращает ("unknown_pipeline", "unknown_build")
    
    Примеры:
        >>> from pathlib import Path
        >>> parse_filename(Path("207920-26884038-logs_content.lst"))
        ('207920', '26884038')
        >>> parse_filename(Path("123-456-test.lst"))
        ('123', '456')
        >>> parse_filename(Path("invalid.lst"))
        ('unknown_pipeline', 'unknown_build')
    """
    stem = path.stem  # без расширения
    parts = stem.split("-")
    if len(parts) < 3:
        return "unknown_pipeline", "unknown_build"

    pipeline_id = parts[0]
    build_id = parts[1]
    return pipeline_id, build_id


def resolve_dataset_paths(cfg: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    """
    Извлекает основные пути из конфигурации dataset.
    
    Извлекает следующие пути из config.yaml:
    - prepared: директория с подготовленными файлами
    - drain: директория с Drain3 файлами
    - vocab: директория со словарем
    - dataset: директория с разделенными данными (train/val/test)
    
    Args:
        cfg: Конфигурация из config.yaml (dict)
    
    Returns:
        Словарь с путями:
        {
            "prepared": Path | None,
            "drain": Path | None,
            "vocab": Path | None,
            "dataset": Path | None,
        }
    """
    dataset_cfg = cfg.get("dataset", {})
    
    def _get_path(key: str) -> Optional[Path]:
        value = dataset_cfg.get(key)
        if isinstance(value, str):
            return Path(value)
        return None
    
    return {
        "prepared": _get_path("prepared"),
        "drain": _get_path("drain"),
        "vocab": _get_path("vocab"),
        "dataset": _get_path("dataset"),
    }


def get_input_files_from_config(cfg: Dict[str, Any], for_split: bool = False) -> List[Path]:
    """
    Извлекает входные файлы из конфигурации.
    
    Логика зависит от параметра for_split:
    - Если for_split=True: использует dataset.drain_chunks (один файл)
    - Если for_split=False: использует dataset.vocab_list или dataset.dataset_list (список файлов)
    
    Args:
        cfg: Конфигурация из config.yaml
        for_split: Если True, использует dataset.drain_chunks для разделения датасета
                   Если False, использует vocab_list для построения словаря
    
    Returns:
        Список путей к входным файлам
    """
    from modules.logger import get_logger
    
    log = get_logger(__name__)
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


def get_vocab_path_from_config(cfg: Union[Config, Dict[str, Any]]) -> Optional[Path]:
    """
    Получает путь к файлу словаря из конфигурации.
    
    Извлекает путь из dataset.vocab и добавляет event_vocab.json.
    Формат: dataset.vocab/event_vocab.json
    
    Args:
        cfg: Конфигурация (Config объект или dict)
    
    Returns:
        Path к файлу event_vocab.json или None, если не указан или файл не существует
    
    Примеры:
        >>> from modules.config import get_config
        >>> cfg = get_config()
        >>> vocab_path = get_vocab_path_from_config(cfg)
        >>> if vocab_path:
        ...     print(f"Vocab path: {vocab_path}")
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(cfg, 'get'):
        # Config объект
        vocab_dir = cfg.get("dataset.vocab")
    else:
        # dict
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
        vocab_dir = dataset_cfg.get("vocab")
    
    if not vocab_dir:
        return None
    
    vocab_path = Path(vocab_dir) / "event_vocab.json"
    return vocab_path if vocab_path.exists() else None


def get_dataset_path_from_config(cfg: Union[Config, Dict[str, Any]], key: str) -> Optional[Path]:
    """
    Получает путь из секции dataset конфигурации.
    
    Универсальная функция для извлечения путей из dataset секции.
    
    Args:
        cfg: Конфигурация (Config объект или dict)
        key: Ключ в секции dataset (например, "dataset", "prepared", "drain")
    
    Returns:
        Path или None, если не указан
    
    Примеры:
        >>> from modules.config import get_config
        >>> cfg = get_config()
        >>> dataset_dir = get_dataset_path_from_config(cfg, "dataset")
        >>> prepared_dir = get_dataset_path_from_config(cfg, "prepared")
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(cfg, 'get'):
        # Config объект
        value = cfg.get(f"dataset.{key}")
    else:
        # dict
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
        value = dataset_cfg.get(key)
    
    if isinstance(value, str):
        return Path(value)
    return None


def get_window_size_from_config(cfg: Union[Config, Dict[str, Any]], default: int = 64) -> int:
    """
    Получает размер окна из конфигурации.
    
    Извлекает значение из dataset.windows.size.
    
    Args:
        cfg: Конфигурация (Config объект или dict)
        default: Значение по умолчанию, если не указано (по умолчанию 64)
    
    Returns:
        Размер окна (int)
    
    Примеры:
        >>> from modules.config import get_config
        >>> cfg = get_config()
        >>> window_size = get_window_size_from_config(cfg)
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(cfg, 'get'):
        # Config объект
        window_size = cfg.get("dataset.windows.size", default)
    else:
        # dict
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
        windows_cfg = dataset_cfg.get("windows", {})
        window_size = windows_cfg.get("size", default) if isinstance(windows_cfg, dict) else default
    
    return int(window_size)


def get_inputs_from_drain_list(cfg: Union[Config, Dict[str, Any]]) -> List[Path]:
    """
    Получает список входных файлов из dataset.drain_list.
    
    Используется для получения списка JSONL файлов (train.jsonl, val.jsonl, test.jsonl).
    
    Args:
        cfg: Конфигурация (Config объект или dict)
    
    Returns:
        Список путей к входным файлам
    
    Примеры:
        >>> from modules.config import get_config
        >>> cfg = get_config()
        >>> inputs = get_inputs_from_drain_list(cfg)
    """
    from modules.logger import get_logger
    
    log = get_logger(__name__)
    
    # Поддерживаем как Config объект, так и dict
    if hasattr(cfg, 'get'):
        # Config объект
        drain_list = cfg.get("dataset.drain_list", [])
    else:
        # dict
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
        drain_list = dataset_cfg.get("drain_list", [])
    
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


def get_train_val_paths_from_config(cfg: Union[Config, Dict[str, Any]]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Получает пути к train.pt, val.pt и vocab из конфигурации.
    
    Извлекает пути из:
    - train.pt: dataset.dataset/train.pt
    - val.pt: dataset.dataset/val.pt (опционально, если файл существует)
    - vocab: dataset.vocab/event_vocab.json
    
    Args:
        cfg: Конфигурация (Config объект или dict)
    
    Returns:
        Кортеж (train_pt, val_pt, vocab_path)
        Если файл не найден, соответствующий элемент будет None
    """
    # Поддерживаем как Config объект, так и dict
    if hasattr(cfg, 'get_dataset_config'):
        # Config объект
        dataset_cfg = cfg.get_dataset_config()
    else:
        # dict
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    
    # train.pt: dataset.dataset/train.pt
    dataset_dir_str = dataset_cfg.get("dataset")
    train_pt = None
    val_pt = None
    if dataset_dir_str:
        dataset_dir = Path(dataset_dir_str)
        train_pt = dataset_dir / "train.pt"
        if not train_pt.exists():
            train_pt = None
        
        # val.pt: dataset.dataset/val.pt (опционально)
        val_pt_candidate = dataset_dir / "val.pt"
        if val_pt_candidate.exists():
            val_pt = val_pt_candidate
    
    # vocab: dataset.vocab/event_vocab.json
    vocab_path = get_vocab_path_from_config(cfg)
    
    return train_pt, val_pt, vocab_path

