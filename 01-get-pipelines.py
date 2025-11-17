#!/usr/bin/env python3
"""
Получить список Azure DevOps (TFS) pipelines и сохранить их в pipelines.csv.
Кэширует ответ в pipelines.json, чтобы не дергать API лишний раз.
"""

import os
import base64
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import cast, Any

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
API_URL = f"{TFS_API_URL}/{TFS_ORG}/{TFS_PROJECT}/_apis/pipelines"

def fetch_pipelines() -> dict | None:
    """Вернуть распарсенный JSON с API либо None при ошибке."""
    headers = {
        "Authorization": f"Basic {base64.b64encode(f':{TFS_PAT}'.encode()).decode()}",
        "Content-Type": "application/json",
    }
    try:
        rsp = requests.get(API_URL, headers=headers, verify=False, timeout=30)
        log.debug("HTTP %s", rsp.status_code)
        rsp.raise_for_status()
        return cast(dict[str, Any], rsp.json())
    except requests.RequestException as exc:
        log.error("Ошибка запроса: %s", exc)
        return None

def load_cache() -> dict | None:
    cache = Path("pipelines.json")
    if cache.exists():
        try:
            return cast(dict[str, Any], json.loads(cache.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("Не удалось прочитать кэш: %s", exc)
    return None

def save_cache(data: dict) -> None:
    Path("pipelines.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(pipelines: list[dict], prefix: str | None) -> None:
    """Сохраняем pipelines в CSV и красивый текстовый лист."""
    if not pipelines:
        log.warning("Нет данных для сохранения")
        return

    # DataFrame из списка словарей
    df = pd.DataFrame(pipelines)[["id", "name", "folder"]].rename(
        columns={"id": "ID", "name": "Name", "folder": "Folder"}
    )

    # 1. чистый CSV
    df.to_csv("pipelines.csv", index=False, encoding="utf-8")
    log.info("Сохранено %s записей в pipelines.csv", len(df))

    # 2. форматированный лист для печати (широкий вывод)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        Path("pipelines.lst").write_text(str(df), encoding="utf-8")

# --- main --------------------------------------------------------------------
def main(prefix: str | None = None) -> None:
    log.debug("prefix %s", prefix)
    start = datetime.now()

    data = load_cache() or fetch_pipelines()
    if not data:
        log.error("Не удалось получить данные")
        sys.exit(1)

    # обновляем кэш только если данные были получены через API
    if not load_cache():
        save_cache(data)

    pipelines = data.get("value", [])
    #log.debug("pipelines input %s", pipelines)

    if prefix:
        pipelines = [p for p in pipelines if (p.get("folder") or "").startswith(prefix)]
    #log.debug("pipelines output %s", pipelines)

    write_csv(pipelines, prefix)
    log.info("Завершено за %.2f с", (datetime.now() - start).total_seconds())

# --- entry-point -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт Azure DevOps pipelines в CSV")
    parser.add_argument("prefix", nargs="?", help="Префикс папки (folder)")
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
    main(args.prefix)
