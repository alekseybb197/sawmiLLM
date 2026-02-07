#!/usr/bin/env python3
"""
Получить логи Azure DevOps (TFS) pipeline run и сохранить их в logs.csv.
Загрузить содержимое логов и сохранить в отдельные файлы.
Кэширует ответ в logs.json, чтобы не дергать API лишний раз.
"""

import os
import base64
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import cast, Any
import re

import requests
import urllib3
import argparse
import pandas as pd

# --- logging -----------------------------------------------------------------
def set_log_level(level: str) -> None:
    """Установить уровень логирования глобально."""
    level = level.upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        sys.exit(f"Недопустимый уровень логирования: {level}")
    logging.getLogger().setLevel(level)
    # при необходимости можно пере-настроить handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)

# 1. create logger first
logging.basicConfig(
    level=logging.INFO,          # will be overridden by set_log_level later
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)   # <-- now log exists

# --- disable SSL warnings ----------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- env shortcuts -----------------------------------------------------------
TFS_PAT        = os.getenv("TFS_PAT_TOKEN")
TFS_API_URL    = os.getenv("TFS_API_URL")
TFS_ORG        = os.getenv("TFS_ORGANIZATION")
TFS_PROJECT    = os.getenv("TFS_PROJECT")

MISSING = [v for v in ("TFS_PAT_TOKEN", "TFS_API_URL", "TFS_ORGANIZATION", "TFS_PROJECT")
           if not os.getenv(v)]
if MISSING:
    log.error("Отсутствуют обязательные переменные окружения: %s", ", ".join(MISSING))
    sys.exit(1)

# --- core --------------------------------------------------------------------
def get_logs_api_url(pipeline_id: int, run_id: int) -> str:
    """Вернуть URL API для получения логов билда."""
    return f"{TFS_API_URL}/{TFS_ORG}/{TFS_PROJECT}/_apis/pipelines/{pipeline_id}/runs/{run_id}/logs"

def get_log_content_api_url(run_id: int, log_num: int) -> str:
    """Вернуть URL API для получения содержимого лога."""
    return f"{TFS_API_URL}/{TFS_ORG}/{TFS_PROJECT}/_apis/build/builds/{run_id}/logs/{log_num}"

def make_api_request(url: str, return_text: bool = False) -> dict | str | None:
    """Выполнить API запрос и вернуть JSON/текст или None при ошибке."""
    headers = {
        "Authorization": f"Basic {base64.b64encode(f':{TFS_PAT}'.encode()).decode()}",
        "Content-Type": "application/json",
    }
    
    try:
        log.debug("Запрос к URL: %s", url)
        rsp = requests.get(url, headers=headers, verify=False, timeout=30)
        log.debug("HTTP %s", rsp.status_code)
        rsp.raise_for_status()
        
        if return_text:
            # Для текстового содержимого возвращаем как есть с правильной кодировкой
            rsp.encoding = 'utf-8'  # Принудительно устанавливаем UTF-8
            return rsp.text
        
        # Пытаемся распарсить как JSON
        try:
            return cast(dict[str, Any], rsp.json())
        except ValueError:
            # Если не JSON, возвращаем текст с правильной кодировкой
            rsp.encoding = 'utf-8'
            return rsp.text
    except requests.RequestException as exc:
        log.error("Ошибка запроса: %s", exc)
        return None

def fetch_logs(pipeline_id: int, run_id: int) -> dict | None:
    """Вернуть распарсенный JSON с логами билда либо None при ошибке."""
    url = get_logs_api_url(pipeline_id, run_id)
    result = make_api_request(url)
    if isinstance(result, dict):
        return result
    return None

def fetch_log_content(run_id: int, log_num: int) -> str | None:
    """Вернуть содержимое лога либо None при ошибке."""
    url = get_log_content_api_url(run_id, log_num)
    result = make_api_request(url, return_text=True)
    if isinstance(result, str):
        return result
    return None

def get_file_prefix(pipeline_id: int, run_id: int) -> str:
    """Получить префикс для имен файлов."""
    return f"{pipeline_id}-{run_id}-"

def load_cache(pipeline_id: int, run_id: int) -> dict | None:
    """Загрузить данные из кэш-файла."""
    prefix = get_file_prefix(pipeline_id, run_id)
    cache = Path(f"{prefix}logs.json")
    if cache.exists():
        try:
            return cast(dict[str, Any], json.loads(cache.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("Не удалось прочитать кэш: %s", exc)
    return None

def save_cache(data: dict, pipeline_id: int, run_id: int) -> None:
    """Сохранить данные в кэш-файл."""
    prefix = get_file_prefix(pipeline_id, run_id)
    Path(f"{prefix}logs.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def extract_log_number_from_url(log_url: str) -> str:
    """Извлечь номер лога из URL."""
    return log_url.split('/')[-1]

def normalize_line_endings(text: str) -> str:
    """Нормализовать символы конца строк: заменить CR+LF и CR на LF."""
    if text is None:
        return ""
    # Заменяем CR+LF и CR на LF
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    return text

def has_timestamp_and_section(line: str) -> tuple[bool, str]:
    """
    Проверить, содержит ли строка таймстамп и секцию.
    Возвращает (has_timestamp, section_title)
    """
    # Паттерн для таймстампа: 2025-10-22T11:28:59.7647794Z
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+'
    
    # Проверяем наличие таймстампа
    if not re.match(timestamp_pattern, line):
        return False, ""
    
    # Ищем секцию в формате ##[section]...
    section_match = re.search(r'##\[section\](.+)', line)
    if section_match:
        section_title = section_match.group(1).strip()
        return True, section_title
    
    return True, ""

def process_log_content(content: str) -> tuple[bool, str, str]:
    """
    Обработать содержимое лога.
    Возвращает (should_include, section_title, processed_content)
    """
    if not content:
        return False, "", ""
    
    lines = content.split('\n')
    if not lines:
        return False, "", ""
    
    # Проверяем первую строку на наличие таймстампа
    first_line = lines[0]
    has_timestamp, section_title = has_timestamp_and_section(first_line)
    
    if not has_timestamp:
        return False, "", ""
    
    return True, section_title, content

def download_logs_content(logs_data: dict, pipeline_id: int, run_id: int) -> None:
    """Загрузить содержимое всех логов и сохранить в файлы."""
    logs = logs_data.get("logs", [])
    if not logs:
        log.warning("Нет логов для загрузки")
        return

    prefix = get_file_prefix(pipeline_id, run_id)
    log_file = Path(f"{prefix}logs_content.txt")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        for log_entry in logs:
            log_url = log_entry.get("url", "")
            if not log_url:
                continue
                
            log_num_str = extract_log_number_from_url(log_url)
            try:
                log_num = int(log_num_str)
            except ValueError:
                log.warning("Не удалось преобразовать номер лога в int: %s", log_num_str)
                continue
            log_id = log_entry.get("id", "unknown")
            
            log.info("Загрузка лога %s", log_num)
            
            # Загружаем содержимое лога
            log_content = fetch_log_content(run_id, log_num)
            if not log_content:
                continue
            
            # Нормализуем символы конца строк
            normalized_content = normalize_line_endings(log_content)
            
            # Обрабатываем содержимое лога
            should_include, section_title, processed_content = process_log_content(normalized_content)
            
            if not should_include:
                log.debug("Пропущен лог %s: отсутствует таймстамп в первой строке", log_num)
                continue
            
            # Добавляем заголовок в файл
            section_part = f"##[section]{section_title} === " if section_title else ""
            header = f"\n======= {section_part}Pipeline: {pipeline_id} === Run: {run_id} === Log: {log_id} === {log_url}\n"
            
            f.write(header)
            f.write(processed_content)
            f.write("\n")
    
    log.info("Содержимое логов сохранено в %s", log_file)

def write_logs_csv(logs_data: dict, pipeline_id: int, run_id: int) -> None:
    """Сохраняем логи в CSV и текстовый файл."""
    logs = logs_data.get("logs", [])
    if not logs:
        log.warning("Нет логов для сохранения")
        return

    prefix = get_file_prefix(pipeline_id, run_id)

    # Подготовка данных для DataFrame
    processed_logs = []
    for log_entry in logs:
        processed_logs.append({
            "id": log_entry.get("id"),
            "lineCount": log_entry.get("lineCount", 0),
            "createdOn": log_entry.get("createdOn"),
            "url": log_entry.get("url", "N/A"),
            "lastChangedOn": log_entry.get("lastChangedOn", "N/A")
        })

    # DataFrame из списка словарей
    df = pd.DataFrame(processed_logs).rename(
        columns={
            "id": "ID", 
            "lineCount": "Lines", 
            "createdOn": "Created", 
            "url": "URL",
            "lastChangedOn": "Last Changed"
        }
    )

    # 1. чистый CSV
    df.to_csv(f"{prefix}logs.csv", index=False, encoding="utf-8")
    log.info("Сохранено %s записей логов в %slogs.csv", len(df), prefix)

    # 2. форматированный лист для печати
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        Path(f"{prefix}logs.lst").write_text(str(df), encoding="utf-8")

# --- main --------------------------------------------------------------------
def main(pipeline_id: int, run_id: int) -> None:
    """Основная функция для получения логов билда."""
    log.debug("pipeline_id: %s, run_id: %s", pipeline_id, run_id)
    start = datetime.now()

    # Используем кэш если есть
    data = load_cache(pipeline_id, run_id)
    
    if data:
        log.info("Используются кэшированные данные логов")
    else:
        data = fetch_logs(pipeline_id, run_id)
        if data:
            save_cache(data, pipeline_id, run_id)

    if not data:
        log.error("Не удалось получить логи")
        sys.exit(1)

    write_logs_csv(data, pipeline_id, run_id)
    log.info("✅ Успешно получен список логов")
    
    # Загружаем содержимое логов
    download_logs_content(data, pipeline_id, run_id)
    log.info("✅ Успешно загружено содержимое логов")
    
    log.info("Завершено за %.2f с", (datetime.now() - start).total_seconds())

# --- entry-point -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Получить логи Azure DevOps pipeline run")
    
    # Позиционные параметры
    parser.add_argument("pipeline_id", type=int, help="ID пайплайны")
    parser.add_argument("run_id", type=int, help="ID билда (run)")
    
    # Общие параметры
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования (по умолчанию: $LOG_LEVEL или INFO)",
    )
    
    args = parser.parse_args()

    # устанавливаем уровень ДО первого лога
    set_log_level(args.log_level)

    # Вызываем основную функцию
    main(args.pipeline_id, args.run_id)