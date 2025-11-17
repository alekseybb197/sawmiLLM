"""
Модуль для настройки и управления логгированием.
Поддерживает консольный вывод, файловое логгирование и JSON формат.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from modules.config import get_config


class JSONFormatter(logging.Formatter):
    """Форматтер для записи логов в JSON формате."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись лога в JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Добавляем дополнительные поля если они есть
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry, ensure_ascii=False, indent=2)


class Logger:
    """Класс для настройки логгирования с поддержкой множественных форматов."""
    
    def __init__(self, name: str, script_name: Optional[str] = None):
        """
        Инициализация логгера.
        
        Args:
            name: Имя логгера
            script_name: Имя скрипта для создания уникальных файлов логов
        """
        self.name = name
        self.script_name = script_name or name
        self.config = get_config()
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Настраивает логгер с консольным и файловым выводом."""
        # Очищаем существующие обработчики
        self.logger.handlers.clear()
        
        # Устанавливаем уровень логгирования
        log_level = getattr(logging, self.config.get("logging.level", "INFO"))
        self.logger.setLevel(log_level)
        
        # Создаем форматтеры
        console_formatter = logging.Formatter(
            self.config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        json_formatter = JSONFormatter()
        
        # Консольный обработчик
        if self.config.get("logging.console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Файловый обработчик для обычных логов
        if self.config.get("logging.file_output", True):
            self._setup_file_handler(console_formatter)
        
        # JSON обработчик
        if self.config.get("logging.json_output", True):
            self._setup_json_handler(json_formatter)
    
    def _setup_file_handler(self, formatter: logging.Formatter) -> None:
        """Настраивает файловый обработчик логов."""
        logs_dir = self.config.get("dataset.logs_dir", "./logs")
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.script_name}_{timestamp}.log"
        log_path = os.path.join(logs_dir, log_filename)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Файловое логгирование настроено: {log_path}")
    
    def _setup_json_handler(self, formatter: JSONFormatter) -> None:
        """Настраивает JSON обработчик логов."""
        logs_dir = self.config.get("dataset.logs_dir", "./logs")
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{self.script_name}_{timestamp}.json"
        json_path = os.path.join(logs_dir, json_filename)
        
        json_handler = logging.FileHandler(json_path, encoding='utf-8')
        json_handler.setFormatter(formatter)
        self.logger.addHandler(json_handler)
        
        self.logger.info(f"JSON логгирование настроено: {json_path}")
    
    def log_api_request(self, request_data: Dict[str, Any]) -> None:
        """Логирует запрос к API."""
        extra_data = {"type": "api_request", "data": request_data}
        self.logger.info("API запрос", extra={"extra_data": extra_data})
    
    def log_api_response(self, response_data: Dict[str, Any]) -> None:
        """Логирует ответ API."""
        extra_data = {"type": "api_response", "data": response_data}
        self.logger.info("API ответ", extra={"extra_data": extra_data})
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Логирует метрики."""
        extra_data = {"type": "metrics", "data": metrics}
        self.logger.info("Метрики", extra={"extra_data": extra_data})
    
    def log_data_quality(self, quality_data: Dict[str, Any]) -> None:
        """Логирует данные о качестве."""
        extra_data = {"type": "data_quality", "data": quality_data}
        self.logger.info("Качество данных", extra={"extra_data": extra_data})
    
    def get_logger(self) -> logging.Logger:
        """Возвращает настроенный логгер."""
        return self.logger


def get_logger(name: str, script_name: Optional[str] = None) -> logging.Logger:
    """
    Создает и возвращает настроенный логгер.
    
    Args:
        name: Имя логгера
        script_name: Имя скрипта для файлов логов
        
    Returns:
        Настроенный логгер
    """
    logger_instance = Logger(name, script_name)
    return logger_instance.get_logger()
