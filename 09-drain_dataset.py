#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drain_dataset.py
======================
Многоэтапная подготовка датасета для LogBERT:

1️⃣  --init-drain   → строит drain_state.json и drain_templates.json
2️⃣  --extend-drain   → добавляет новые логи к существующему Drain state
3️⃣  --encode       → кодирует логи в event_id и сохраняет в drain_sequences.jsonl и drain_chunks.jsonl
4️⃣  --append       → добавляет новые логи в существующие drain_sequences.jsonl и drain_chunks.jsonl

Все файлы сохраняются в директорию dataset.drain из config.yaml.

Файлы:
  • {drain_dir}/drain_state.json        — бинарное состояние Drain3
  • {drain_dir}/drain_templates.json    — человекочитаемые шаблоны E[id] -> template
  • {drain_dir}/drain_sequences.jsonl   — полные последовательности событий для каждого билда
  • {drain_dir}/drain_chunks.jsonl      — чанки для разделения на train/val/test (используется в 04_build_vocab.py)
  • {drain_dir}/drain_mapping.jsonl     — mapping event_id → оригинальные строки

Примечание: Разделение на train/val/test выполняется в 04_build_vocab.py --split-dataset
"""
import os
import re
import json
import random
import hashlib

import signal
import sys
import time
from datetime import datetime

from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Optional
from tqdm import tqdm
import typer
from drain3 import TemplateMiner # type: ignore[import-untyped]
from drain3.file_persistence import FilePersistence # type: ignore[import-untyped]
from drain3.template_miner_config import TemplateMinerConfig # type: ignore[import-untyped]

from modules.config import get_config
from modules.logger import get_logger


# ---------------------------------------------------------
# CLI (Typer)
# ---------------------------------------------------------
app = typer.Typer(
    name="drain_dataset",
    help="Drain3 датасет конструктор: инициализация, расширение, кодирование и добавление данных",
    add_completion=False,
)


# ---------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------
def load_config() -> dict:
    cfg = get_config()
    dataset_cfg = cfg.get("dataset", {})
    drain_cfg = dataset_cfg.get("drain3_config", {})

    # Базовая директория для всех drain-файлов из dataset.drain
    drain_dir = Path(dataset_cfg.get("drain", "./dataset/drain"))
    drain_dir.mkdir(parents=True, exist_ok=True)

    # Настройки секций
    section_markers_cfg = dataset_cfg.get("section_markers", {})
    
    return {
        # пути
        "drain_dir": drain_dir,
        "prepared_dir": Path(dataset_cfg.get("prepared", "./dataset/prepared")),
        # Все drain-файлы сохраняются в drain_dir
        "state_file": drain_dir / "drain_state.json",
        "templates_file": drain_dir / "drain_templates.json",
        "sequences_file": drain_dir / "drain_sequences.jsonl",

        # чанки
        "min_seq_len": int(dataset_cfg.get("min_seq_len", 5)),
        "max_chunk_len": int(dataset_cfg.get("max_chunk_len", 512)),
        "chunk_stride": int(dataset_cfg.get("chunk_stride", 256)),

        # секции
        "section_markers": {
            "use_azure_sections": bool(section_markers_cfg.get("use_azure_sections", True)),
            "use_tfs_sections": bool(section_markers_cfg.get("use_tfs_sections", True)),
            "tfs_regex": section_markers_cfg.get("tfs_regex", r"<TFS_Section>\s*(.+)"),
        },

        # Drain3
        "drain3_config": drain_cfg,
    }


# ---------------------------------------------------------
# Разметка секций
# ---------------------------------------------------------
def parse_sections_from_lines(lines: List[str], sect_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    use_azure = sect_cfg.get("use_azure_sections", True)
    use_tfs = sect_cfg.get("use_tfs_sections", True)
    tfs_pat = re.compile(sect_cfg.get("tfs_regex", r"<TFS_Section>\s*(.+)"))
    sections: Dict[str, List[str]] = {}
    current_name = ""
    current_lines: List[str] = []

    def _push():
        nonlocal current_name, current_lines
        if current_name and current_lines:
            name = current_name
            idx = 2
            while name in sections:
                name = f"{current_name}#{idx}"
                idx += 1
            sections[name] = current_lines
        current_name, current_lines = "", []

    for line in lines:
        stripped = line.strip()

        # Azure DevOps
        if use_azure and "##[section]Starting:" in stripped:
            _push()
            current_name = stripped.split("##[section]Starting:")[-1].strip()
            current_lines = []
            continue
        if use_azure and "##[section]Finishing:" in stripped:
            _push()
            continue

        # TFS
        if use_tfs:
            m = tfs_pat.search(stripped)
            if m:
                _push()
                current_name = m.group(1).strip()
                current_lines = []
                continue
        if current_name:
            current_lines.append(stripped)
    _push()
    return sections or {"full": lines}


# ---------------------------------------------------------
# Чанкование
# ---------------------------------------------------------
def chunk_sequence(seq: List[str], max_len: int, stride: int, min_len: int) -> List[List[str]]:
    if len(seq) == 0:
        return []
    if len(seq) <= max_len:
        return [seq] if len(seq) >= min_len else []
    chunks = []
    for i in range(0, len(seq), stride):
        ch = seq[i:i + max_len]
        if len(ch) >= min_len:
            chunks.append(ch)
        if i + max_len >= len(seq):
            break
    return chunks


# ---------------------------------------------------------
# Drain3 init / extend
# ---------------------------------------------------------
def build_drain_state(cfg: dict, logger, extend: bool = False):
    dcfg = cfg["drain3_config"]

    # --- Конфигурация TemplateMiner ---
    tm_cfg = TemplateMinerConfig()

    # Основные параметры дерева Drain
    tm_cfg.drain_depth = int(dcfg.get("depth", 4))
    tm_cfg.drain_sim_threshold = float(dcfg.get("sim_threshold", 0.5))
    tm_cfg.drain_max_children = int(dcfg.get("max_children", 200))
    tm_cfg.drain_max_clusters = int(dcfg.get("max_clusters", 20000))

    # Разделители и шаблон переменных
    tm_cfg.param_str = dcfg.get("param_str", "<*>")
    tm_cfg.param_regex = dcfg.get("param_regex", r"[\w\-\.:/\\]+")
    tm_cfg.extra_delimiters = dcfg.get("extra_delimiters", [])

    # --- Включаем хранение параметров и примеров сообщений ---
    tm_cfg.store_parameters = dcfg.get("store_parameters", False)
    if tm_cfg.store_parameters:                 # 🟢 сохраняет примеры строк для каждого шаблона
        tm_cfg.profiling_enabled = False        # не нужно профилирование
        tm_cfg.snapshot_interval_minutes = 0    # отключаем автоснапшоты
        tm_cfg.snapshot_compress_state = False  # не архивировать снапшоты

    # --- Создаем директорию для снапшотов ---
    snapshot_dir = cfg["drain_dir"] / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    main_state_file = cfg["state_file"]
    
    # --- Инициализация Drain3: работа в памяти для ускорения ---
    # Загружаем начальное состояние если есть
    if extend and main_state_file.exists():
        # Загружаем из файла, затем переключаемся на работу в памяти
        temp_persistence = FilePersistence(str(main_state_file))
        miner = TemplateMiner(persistence_handler=temp_persistence)
        logger.info(f"Загружено состояние из {main_state_file} ({len(miner.drain.clusters)} шаблонов)")
        # Переключаемся на работу в памяти (None persistence)
        miner.persistence_handler = None
    else:
        # Создаем новый miner без persistence - работа в памяти
        miner = TemplateMiner(persistence_handler=None, config=tm_cfg)

    mode = "расширение" if extend else "инициализация"
    logger.info(f"=== Drain {mode.upper()} ===")
    print(f"🚀 Запуск Drain {mode} дерева...")

    files = sorted(cfg["prepared_dir"].glob("*.lst"))
    if not files:
        logger.error(f"❌ Нет файлов в {cfg['prepared_dir']}")
        return

    # --- параметры мониторинга ---
    total_lines = 0
    new_clusters_start = len(miner.drain.clusters)
    progress_interval = 50000     # каждые 50k строк логируем прогресс
    save_interval = 300000        # каждые 300k строк сохраняем state
    last_save_time = time.time()
    interrupted = False

    def handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n⚠️ Прерывание по Ctrl-C. Идёт безопасное сохранение Drain state...")
    signal.signal(signal.SIGINT, handle_sigint)

    def save_state_to_file(miner_instance: TemplateMiner, target_file: Path, reason: str) -> None:
        """Сохраняет состояние miner в указанный файл."""
        # Временно переключаемся на FilePersistence для сохранения
        original_handler = miner_instance.persistence_handler
        temp_persistence = FilePersistence(str(target_file))
        miner_instance.persistence_handler = temp_persistence
        try:
            miner_instance.save_state(snapshot_reason=reason)
        finally:
            # Возвращаем обратно на None для работы в памяти
            miner_instance.persistence_handler = original_handler

    try:
        # Используем tqdm для визуального прогресса по файлам
        for path in tqdm(files, desc="Drain update"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if interrupted:
                        raise KeyboardInterrupt
                    line = line.strip()
                    if not line:
                        continue
                    result = miner.add_log_message(line)
                    total_lines += 1
                    if total_lines % progress_interval == 0:
                        clusters_now = len(miner.drain.clusters)
                        new_clusters = clusters_now - new_clusters_start
                        elapsed = time.time() - last_save_time
                        # Логируем в файл, но не в консоль (tqdm займёт консоль)
                        logger.info(f"[Drain] {total_lines:,} строк, {clusters_now} шаблонов (+{new_clusters}), {elapsed:.1f}s с последнего сохранения")
                    if total_lines % save_interval == 0:
                        # Сохраняем снапшот в отдельный файл
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snapshot_file = snapshot_dir / f"drain_state_snapshot_{timestamp}_lines_{total_lines}.json"
                        save_state_to_file(miner, snapshot_file, f"checkpoint-{total_lines}")
                        # Сообщение о сохранении только в файловый лог
                        logger.info(f"[Drain] Снапшот сохранён: {snapshot_file.name} ({total_lines:,} строк)")
                        last_save_time = time.time()

        # Финальное сохранение в основной файл
        save_state_to_file(miner, main_state_file, "extend-drain" if extend else "init-drain")
        # Итоговое сообщение в файловый лог
        logger.info(f"💾 Финальное состояние сохранено в {main_state_file.name}")
        
        # --- Обновляем шаблоны сразу после сохранения состояния ---
        clusters_final = len(miner.drain.clusters)
        templates = {f"E{c.cluster_id}": c.get_template() for c in miner.drain.clusters}
        with open(cfg["templates_file"], "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        logger.info(f"📝 Шаблоны обновлены после сохранения: {clusters_final} шаблонов")

    except KeyboardInterrupt:
        # Сохраняем состояние при прерывании
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupt_snapshot = snapshot_dir / f"drain_state_snapshot_interrupt_{timestamp}_lines_{total_lines}.json"
        save_state_to_file(miner, interrupt_snapshot, "interrupt")
        # Также сохраняем в основной файл для восстановления
        save_state_to_file(miner, main_state_file, "interrupt")
        logger.warning(f"🚨 Drain state сохранён после прерывания ({total_lines:,} строк обработано)")
        logger.warning(f"🚨 Снапшот прерывания: {interrupt_snapshot.name}")
        # Обновляем шаблоны после прерывания
        clusters_interrupt = len(miner.drain.clusters)
        templates = {f"E{c.cluster_id}": c.get_template() for c in miner.drain.clusters}
        with open(cfg["templates_file"], "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        logger.info(f"📝 Шаблоны обновлены после прерывания: {clusters_interrupt} шаблонов")
        print("\n✅ Состояние сохранено. Завершение процесса.")
        sys.exit(0)

    # --- финальная сводка ---
    clusters_final = len(miner.drain.clusters)
    logger.info(f"Drain {mode} завершено: обработано {total_lines:,} строк, всего шаблонов {clusters_final}")

    print(f"\n✅ Drain {mode} завершено ({clusters_final} шаблонов, {total_lines:,} строк).")


# ---------------------------------------------------------
# Кодирование логов (encode/append)
# ---------------------------------------------------------
def encode_logs(cfg: dict, logger, append_mode: bool = False):
    # --- Функция для сохранения состояния ---
    def save_state_to_file(miner_instance: TemplateMiner, target_file: Path, reason: str) -> None:
        """Сохраняет состояние miner в указанный файл."""
        original_handler = miner_instance.persistence_handler
        temp_persistence = FilePersistence(str(target_file))
        miner_instance.persistence_handler = temp_persistence
        try:
            miner_instance.save_state(snapshot_reason=reason)
        finally:
            miner_instance.persistence_handler = original_handler
    
    # --- Инициализация TemplateMiner ---
    persistence = FilePersistence(str(cfg["state_file"]))
    miner = TemplateMiner(persistence_handler=persistence)
    sect_cfg = cfg["section_markers"]
    
    # Отслеживаем количество кластеров на старте
    initial_cluster_count = len(miner.drain.clusters)
    logger.info(f"Начальное количество кластеров: {initial_cluster_count}")

    # --- Создание файла mapping.jsonl для сохранения mapping event_id → оригинальные строки
    mapping_file = cfg["drain_dir"] / "drain_mapping.jsonl"
    map_mode = "a" if append_mode and mapping_file.exists() else "w"
    mapping_out = open(mapping_file, map_mode, encoding="utf-8")

    # --- Обработка файлов ---
    prepared_files = sorted(cfg["prepared_dir"].glob("*.lst"))
    if not prepared_files:
        logger.error(f"❌ Нет файлов в {cfg['prepared_dir']}")
        return

    # --- Загрузка существующих чанков для дедупликации из drain_chunks.jsonl ---
    seen_hashes = set()
    chunks_file = cfg["drain_dir"] / "drain_chunks.jsonl"
    if append_mode and chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = f"{rec['pipeline_id']}|{rec['section_name']}|{' '.join(rec['event_seq'])}"
                    seen_hashes.add(hashlib.sha1(key.encode()).hexdigest())
                except Exception:
                    continue
        logger.info(f"Загружено {len(seen_hashes)} существующих чанков для дедупликации")

    # --- Обработка новых чанков ---
    new_records = []
    per_build = []
    dropped = 0

    for path in tqdm(prepared_files, desc="Encode Logs"):
        parts = path.stem.split('-')
        pipeline_id = parts[0]
        build_id = parts[1] if len(parts) > 1 else "unknown"
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        sections = parse_sections_from_lines(lines, sect_cfg)
        build_seq = []
        # --- Обработка секций ---
        for section_name, lines in sections.items():
            events = []
            # --- Обработка строк секции ---
            for idx, line in enumerate(lines):
                res = miner.add_log_message(line)
                eid = f"E{res['cluster_id']}"
                events.append(eid)
                build_seq.append(eid)
                # --- Записываем mapping ---
                cluster = miner.drain.id_to_cluster.get(res["cluster_id"])
                template = cluster.get_template() if cluster else "<unknown>"
                mapping_out.write(json.dumps({
                    "pipeline_id": pipeline_id,
                    "build_id": build_id,
                    "section_name": section_name,
                    "event_id": eid,
                    "template": template,
                    "original_line": line,
                    "file": str(path),
                    "line_index": idx
                }, ensure_ascii=False) + "\n")
            
            # --- Чанкование и сохранение чанков для последующего разделения ---
            for ch in chunk_sequence(events, cfg["max_chunk_len"], cfg["chunk_stride"], cfg["min_seq_len"]):
                # --- Дедупликация ---
                key = f"{pipeline_id}|{section_name}|{' '.join(ch)}"
                h = hashlib.sha1(key.encode()).hexdigest()
                if h in seen_hashes:
                    dropped += 1
                    continue
                seen_hashes.add(h)
                # Сохраняем чанки в new_records для последующего сохранения в drain_chunks.jsonl
                # (разделение на train/val/test будет выполняться в 04_build_vocab.py)
                new_records.append({
                    "pipeline_id": pipeline_id,
                    "build_id": build_id,
                    "section_name": section_name,
                    "event_seq": ch,
                    "status": "success",
                })
        per_build.append({
            "pipeline_id": pipeline_id,
            "build_id": build_id,
            "event_seq": build_seq,
            "status": "success",
        })

    # Сохраняем полные последовательности билдов в drain_sequences.jsonl
    mode = "a" if append_mode and cfg["sequences_file"].exists() else "w"
    with open(cfg["sequences_file"], mode, encoding="utf-8") as out:
        for rec in per_build:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Сохраняем чанки в отдельный файл для последующего разделения в 04_build_vocab.py
    chunks_file = cfg["drain_dir"] / "drain_chunks.jsonl"
    chunks_mode = "a" if append_mode and chunks_file.exists() else "w"
    with open(chunks_file, chunks_mode, encoding="utf-8") as out:
        for rec in new_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Добавлено {len(new_records)} чанков, отброшено дублей: {dropped}")
    logger.info(f"📝 Полные последовательности сохранены в drain_sequences.jsonl")
    logger.info(f"📝 Чанки сохранены в {chunks_file.name}")
    logger.info(f"💡 Для разделения чанков на train/val/test используйте: 04_build_vocab.py --split-dataset --inputs {chunks_file}")

    # --- Проверяем, были ли созданы новые кластеры ---
    final_cluster_count = len(miner.drain.clusters)
    new_clusters = final_cluster_count - initial_cluster_count
    
    if new_clusters > 0:
        logger.warning(f"⚠️ В процессе encode создано {new_clusters} новых кластеров!")
        logger.warning("⚠️ Сохраняю обновленное состояние в drain_state.json")
        
        # Сохраняем состояние с новыми кластерами
        save_state_to_file(miner, cfg["state_file"], "encode-new-clusters")
        logger.info(f"💾 Обновленное состояние сохранено: {final_cluster_count} кластеров (было {initial_cluster_count})")
    
    # --- Всегда обновляем шаблоны из текущего состояния ---
    clusters = miner.drain.clusters
    templates = {f"E{c.cluster_id}": c.get_template() for c in clusters}
    with open(cfg["templates_file"], "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    
    if new_clusters > 0:
        logger.info(f"📝 Шаблоны обновлены с учетом новых кластеров: {len(templates)} шаблонов (+{new_clusters})")
    else:
        logger.info(f"📝 Шаблоны обновлены: {len(templates)} шаблонов")

    mapping_out.close()
    logger.info(f"💾 drain_mapping.jsonl обновлён ({'append' if append_mode else 'rewrite'})")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
@app.command()
def main(
    init_drain: bool = typer.Option(False, "--init-drain", "-i", help="Построить Drain дерево и шаблоны"),
    extend_drain: bool = typer.Option(False, "--extend-drain", "-e", help="Добавить новые логи к существующему Drain state"),
    encode: bool = typer.Option(False, "--encode", help="Кодировать логи в event_id (без изменения Drain)"),
    append: bool = typer.Option(False, "--append", "-a", help="Добавить новые логи в существующий датасет"),
) -> None:
    """
    Drain3 датасет конструктор для подготовки данных LogBERT.
    
    Все файлы сохраняются в директорию dataset.drain из config.yaml.
    
    Этапы:
      1. --init-drain   → строит drain_state.json и drain_templates.json
      2. --extend-drain → добавляет новые логи к существующему Drain state
      3. --encode       → кодирует логи в event_id и сохраняет в drain_sequences.jsonl и drain_chunks.jsonl
      4. --append       → добавляет новые логи в существующие drain_sequences.jsonl и drain_chunks.jsonl
    
    Примеры:
      python 03_drain_dataset.py --init-drain
      python 03_drain_dataset.py --extend-drain
      python 03_drain_dataset.py --encode
      python 03_drain_dataset.py --append
    """
    # Проверяем, что указан ровно один режим работы
    modes = [init_drain, extend_drain, encode, append]
    if sum(modes) != 1:
        typer.echo("❌ Ошибка: необходимо указать ровно один режим работы (--init-drain, --extend-drain, --encode или --append)", err=True)
        raise typer.Exit(code=1)
    
    cfg = load_config()
    logger = get_logger("drain_dataset", "drain_dataset")
    # drain_dir уже создан в load_config(), но убеждаемся что он существует
    cfg["drain_dir"].mkdir(parents=True, exist_ok=True)

    try:
        if init_drain:
            logger.info("=== Этап INIT-DRAIN ===")
            build_drain_state(cfg, logger, extend=False)
        elif extend_drain:
            logger.info("=== Этап EXTEND-DRAIN ===")
            build_drain_state(cfg, logger, extend=True)
        elif encode:
            logger.info("=== Этап ENCODE ===")
            encode_logs(cfg, logger, append_mode=False)
        elif append:
            logger.info("=== Этап APPEND ===")
            encode_logs(cfg, logger, append_mode=True)

        print("\n✅ Этап завершён успешно.")
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        print("\n⚠️  Прервано пользователем")
        raise typer.Exit(code=130)
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
