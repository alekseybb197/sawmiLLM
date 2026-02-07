#!/usr/bin/env python3
"""
Получить список билдов Azure DevOps (TFS) pipeline и сохранить их в builds.csv.
Кэширует ответ в builds.json, чтобы не дергать API лишний раз.
"""

import os
import base64
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import cast, Any, Optional

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
def get_api_url(pipeline_id: int) -> str:
    """Вернуть URL API для получения билдов пайплайны."""
    return f"{TFS_API_URL}/{TFS_ORG}/{TFS_PROJECT}/_apis/pipelines/{pipeline_id}/runs"

def get_file_prefix(pipeline_id: int, result: str) -> str:
    """Получить префикс для имен файлов."""
    return f"{pipeline_id}-{result}-"

def get_cache_prefix(pipeline_id: int) -> str:
    """Получить префикс для имен файлов."""
    return f"{pipeline_id}-"

def fetch_builds(pipeline_id: int, max_count: Optional[int] = None, min_time: Optional[datetime] = None, max_time: Optional[datetime] = None) -> dict | None:
    """Вернуть распарсенный JSON с API либо None при ошибке."""
    headers = {
        "Authorization": f"Basic {base64.b64encode(f':{TFS_PAT}'.encode()).decode()}",
        "Content-Type": "application/json",
    }
    
    params: dict[str, Any] = {}
    if max_count:
        params['$top'] = max_count
        log.info("Ограничение запроса к API: %s записей", max_count)
    
    # Форматируем даты для API
    if min_time:
        params['minTime'] = str(min_time.isoformat() + 'Z')
    if max_time:
        params['maxTime'] = str(max_time.isoformat() + 'Z')
    
    try:
        url = get_api_url(pipeline_id)
        log.debug("Запрос к URL: %s с параметрами: %s", url, params)
        rsp = requests.get(url, headers=headers, params=params, verify=False, timeout=30)
        log.debug("HTTP %s", rsp.status_code)
        rsp.raise_for_status()
        return cast(dict[str, Any], rsp.json())
    except requests.RequestException as exc:
        log.error("Ошибка запроса: %s", exc)
        return None

def load_cache(pipeline_id: int) -> dict | None:
    prefix = get_cache_prefix(pipeline_id)
    cache = Path(f"{prefix}builds.json")
    if cache.exists():
        try:
            return cast(dict[str, Any], json.loads(cache.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("Не удалось прочитать кэш: %s", exc)
    return None

def save_cache(data: dict, pipeline_id: int) -> None:
    prefix = get_cache_prefix(pipeline_id)
    Path(f"{prefix}builds.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def filter_builds(builds: list[dict], result_filter: str = "succeeded", max_count: Optional[int] = None) -> list[dict]:
    """Фильтрация билдов по состоянию и результату."""
    filtered = []
    
    for build in builds:
        # Фильтруем только completed билды
        state = build.get("state", "").lower()
        if state != "completed":
            log.debug("Пропущен билд %s: state=%s (требуется completed)", build.get("id"), state)
            continue
            
        # Фильтруем по результату
        build_result = build.get("result", "").lower()
        if result_filter == "succeeded" and build_result != "succeeded":
            log.debug("Пропущен билд %s: result=%s (требуется succeeded)", build.get("id"), build_result)
            continue
        elif result_filter == "failed" and build_result != "failed":
            log.debug("Пропущен билд %s: result=%s (требуется failed)", build.get("id"), build_result)
            continue
        elif result_filter == "canceled" and build_result != "canceled":
            log.debug("Пропущен билд %s: result=%s (требуется canceled)", build.get("id"), build_result)
            continue
        
        filtered.append(build)
        log.debug("Добавлен билд %s: state=%s, result=%s", build.get("id"), state, build_result)
        
        # Применяем ограничение по количеству
        if max_count and len(filtered) >= max_count:
            log.info("Достигнуто ограничение в %s записей", max_count)
            break
    
    return filtered

def write_csv(builds: list[dict], pipeline_id: int, result: str, max_count: Optional[int] = None) -> None:
    """Сохраняем builds в CSV и красивый текстовый лист."""
    if not builds:
        log.warning("Нет данных для сохранения")
        return

    prefix = get_file_prefix(pipeline_id, result)

    # Применяем ограничение по количеству записей
    if max_count:
        builds = builds[:max_count]
        log.info("Ограничение вывода в файлы до %s записей", max_count)

    # Подготовка данных для DataFrame
    processed_builds = []
    for build in builds:
        processed_builds.append({
            "id": build.get("id"),
            "name": build.get("name"),
            "state": build.get("state"),
            "result": build.get("result", "N/A"),
            "createdDate": build.get("createdDate"),
            "finishedDate": build.get("finishedDate", "N/A"),
            "pipeline": build.get("pipeline", {}).get("name", "N/A")
        })

    # DataFrame из списка словарей
    df = pd.DataFrame(processed_builds).rename(
        columns={
            "id": "Run ID", 
            "name": "Name", 
            "state": "State", 
            "result": "Result",
            "createdDate": "Created Date",
            "finishedDate": "Finished Date",
            "pipeline": "Pipeline Name"
        }
    )

    # 1. чистый CSV
    df.to_csv(f"{prefix}builds.csv", index=False, encoding="utf-8")
    log.info("Сохранено %s записей в builds.csv", len(df))

    # 2. форматированный лист для печати
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        Path(f"{prefix}builds.lst").write_text(str(df), encoding="utf-8")
    
    # 3. Дополнительный человеко-читаемый вывод (как в bash-скрипте)
    formatted_lines = []
    for build in builds:
        line = f"Run ID: {build.get('id')} | State: {build.get('state')} | Result: {build.get('result', 'N/A')} | Created: {build.get('createdDate')} | Name: {build.get('name')}"
        formatted_lines.append(line)

# --- main --------------------------------------------------------------------
def main(pipeline_id: int, max_count: Optional[int] = None, min_time: Optional[datetime] = None, max_time: Optional[datetime] = None, result_filter: str = "succeeded") -> None:
    log.debug("pipeline_id: %s, max_count: %s, min_time: %s, max_time: %s, result_filter: %s", 
              pipeline_id, max_count, min_time, max_time, result_filter)
    start = datetime.now()

    # Используем кэш только если не указаны параметры фильтрации
    use_cache = not any([max_count, min_time, max_time])
    
    if use_cache:
        data = load_cache(pipeline_id)
        if data:
            log.info("Используются кэшированные данные")
        else:
            data = fetch_builds(pipeline_id, max_count, min_time, max_time)
    else:
        data = fetch_builds(pipeline_id, max_count, min_time, max_time)
    
    if not data:
        log.error("Не удалось получить данные")
        sys.exit(1)

    # обновляем кэш только если данные были получены через API и нет фильтров
    if not load_cache(pipeline_id) and use_cache:
        save_cache(data, pipeline_id)

    builds = data.get("value", [])
    log.info("Получено %s билдов из API", len(builds))

    # Фильтруем билды
    filtered_builds = filter_builds(builds, result_filter, max_count)
    log.info("После фильтрации осталось %s билдов (state=completed, result=%s)", 
             len(filtered_builds), result_filter)

    if not filtered_builds:
        log.warning("Нет билдов, соответствующих критериям фильтрации")
        return

    write_csv(filtered_builds, pipeline_id, result_filter, max_count)
    log.info("Завершено за %.2f с", (datetime.now() - start).total_seconds())

# --- entry-point -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт билдов Azure DevOps pipeline в CSV")
    parser.add_argument("pipeline_id", type=int, help="ID пайплайны")
    parser.add_argument(
        "--max", 
        type=int, 
        dest="max_count",
        help="Максимальное количество билдов для получения и вывода"
    )
    parser.add_argument(
        "--min-time", 
        type=lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')),
        help="Минимальная дата билда в формате ISO (например: 2024-01-01T00:00:00Z)"
    )
    parser.add_argument(
        "--max-time", 
        type=lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')),
        help="Максимальная дата билда в формате ISO (например: 2024-12-31T23:59:59Z)"
    )
    parser.add_argument(
        "--result",
        default="succeeded",
        choices=["succeeded", "failed", "canceled"],
        help="Фильтр по результату выполнения (по умолчанию: succeeded)"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования (по умолчанию: $LOG_LEVEL или INFO)",
    )
    args = parser.parse_args()

    # устанавливаем уровень ДО первого лога
    set_log_level(args.log_level)

    # теперь можно вызывать основную логику
    main(args.pipeline_id, args.max_count, args.min_time, args.max_time, args.result)