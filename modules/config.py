"""
Модуль для загрузки и управления конфигурацией проекта.
Содержит функции для загрузки параметров из config.yaml и их валидации.
"""

import yaml
import os
from typing import Dict, Any, Optional, cast
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Класс для управления конфигурацией проекта."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализация конфигурации.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Загружает конфигурацию из YAML файла."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Файл конфигурации не найден: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            logger.info(f"Конфигурация загружена из {self.config_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение конфигурации по ключу.
        
        Args:
            key: Ключ конфигурации (поддерживает вложенные ключи через точку)
            default: Значение по умолчанию
            
        Returns:
            Значение конфигурации или значение по умолчанию
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_config(self, provider: str) -> Dict[str, Any]:
        """
        Получает конфигурацию API для указанного провайдера.
        
        Args:
            provider: Название провайдера (openai, anthropic, huggingface)
            
        Returns:
            Словарь с конфигурацией API
        """
        return cast(Dict[str, Any], self.get(f"api.{provider}", {}))
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Получает конфигурацию датасета."""
        return cast(Dict[str, Any], self.get("dataset", {}))
    
    def get_filter_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Получает конфигурацию паттернов фильтрации.
        
        Returns:
            Словарь с паттернами фильтрации
        """
        return cast(Dict[str, Dict[str, Any]], self.get("dataset.filter_patterns", {}))
    
    def get_filter_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """
        Получает конкретный паттерн фильтрации.
        
        Args:
            pattern_name: Название паттерна (tfs_timestamp, empty_lines, quality_issues)
            
        Returns:
            Конфигурация паттерна
            
        Raises:
            KeyError: Если паттерн не найден в конфигурации
        """
        patterns = self.get_filter_patterns()
        if pattern_name not in patterns:
            raise KeyError(
                f"Паттерн фильтрации '{pattern_name}' не найден в конфигурации. "
                f"Доступные паттерны: {list(patterns.keys())}. "
                f"Проверьте файл config.yaml в секции dataset.filter_patterns. "
                f"Вызов из: {self._get_caller_info()}"
            )
        return patterns[pattern_name]
    
    def _get_caller_info(self) -> str:
        """Получает информацию о месте вызова для отладки."""
        import inspect
        try:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None and frame.f_back.f_back is not None:
                caller_frame = frame.f_back.f_back
                filename = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno
                function_name = caller_frame.f_code.co_name
                return f"{filename}:{line_number} в функции {function_name}"
        except Exception:
            pass
        return "неизвестное место"
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Получает конфигурацию LoRA."""
        return cast(Dict[str, Any], self.get("lora", {}))
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Получает конфигурацию логгирования."""
        return cast(Dict[str, Any], self.get("logging", {}))
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Получает конфигурацию GPU."""
        return cast(Dict[str, Any], self.get("gpu", {}))
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Получает конфигурацию тестирования."""
        return cast(Dict[str, Any], self.get("testing", {}))
    
    def validate_paths(self) -> None:
        """Проверяет существование необходимых директорий и создает их при необходимости."""
        paths_to_check = [
            self.get("dataset.input_dir", "./tfslogs"),
            self.get("dataset.output_dir", "./processed"),
            self.get("dataset.logs_dir", "./logs"),
            self.get("dataset.checkpoints_dir", "./checkpoints")
        ]
        
        for path in paths_to_check:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Директория проверена/создана: {path}")
    
    def reload(self) -> None:
        """Перезагружает конфигурацию из файла."""
        self.load_config()
        logger.info("Конфигурация перезагружена")


# Глобальный экземпляр конфигурации
config = Config()


def get_config() -> Config:
    """Возвращает глобальный экземпляр конфигурации."""
    return config
