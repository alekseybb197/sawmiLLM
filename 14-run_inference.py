#!/usr/bin/env python3
"""
run_inference.py

Инференс LogBERT на подготовленных логах:

1. Читает config.yaml
2. Загружает Drain3 state, event_vocab, модель, thresholds
3. Для всех файлов вида *-logs_content.lst в dataset.prepared:
   - строки -> Drain -> события (cluster_id -> event_id -> token_id)
   - строит окна длиной windows.size
   - прогоняет через LogBERT
   - считает NLL по каждой позиции
   - сохраняет оценки построчно в *-logs_scored.jsonl
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, cast

import torch
import torch.nn.functional as F
from tqdm import tqdm

from drain3.template_miner import TemplateMiner
from drain3.file_persistence import FilePersistence

from modules.model import UnifiedLogBERT
from modules.utils import ensure_dir
from modules.config import get_config, Config
from modules.logger import get_logger
from modules.sections import parse_sections_from_lines, chunk_sequence
from modules.windowing import section_to_bucket
from modules.paths import parse_filename
from modules.model_utils import (
    resolve_checkpoint_paths,
    get_device_from_config,
    load_model_from_checkpoint,
)


logger = get_logger(__name__)


# ---------------------------
# Drain3
# ---------------------------

def load_drain(state_dir: Path, save_dir: Path) -> TemplateMiner:
    """
    Загружаем сохранённый state Drain3 из state_dir (read-only).
    Drain3 работает в read-only режиме - новые кластеры не создаются.
    
    Args:
        state_dir: Директория с неизменяемым state (dataset.drain)
        save_dir: Директория для сохранения drain_mapping.jsonl, drain_sequences.jsonl, drain_chunks.jsonl
    
    Raises:
        FileNotFoundError: Если drain_state.json не найден
    """
    state_path = state_dir / "drain_state.json"
    
    if not state_path.exists():
        raise FileNotFoundError(
            f"Не найден Drain3 state: {state_path}\n"
            f"Пожалуйста, убедитесь, что файл существует в директории {state_dir}"
        )

    # Загружаем state из read-only директории (dataset.drain)
    state_persistence = FilePersistence(str(state_path))
    template_miner = TemplateMiner(state_persistence)
    template_miner.load_state()

    # Устанавливаем persistence_handler = None для read-only режима
    # Это предотвращает сохранение изменений в drain_state.json
    template_miner.persistence_handler = None
    
    initial_cluster_count = len(template_miner.drain.clusters)
    logger.info("✅ Загружен Drain3 state из %s (read-only, %d кластеров)", state_path, initial_cluster_count)
    logger.info("📁 Drain3 работает в read-only режиме - новые кластеры не создаются")
    
    return template_miner


# ---------------------------
# Vocab / event mapping
# ---------------------------

class EventVocab:
    """
    Обёртка вокруг event_vocab.json.

    Формат словаря:
    {
      "[PAD]": 0,
      "[UNK]": 1,
      "E20020": 2,
      "E20019": 3,
      ...
    }
    """

    def __init__(
        self,
        event_to_id: Dict[str, int],
        pad_id: int,
        unk_id: int,
    ):
        self.event_to_id = event_to_id
        self.pad_id = pad_id
        self.unk_id = unk_id

    @classmethod
    def from_json(cls, path: Path) -> "EventVocab":
        """
        Загружает словарь из JSON файла.
        Формат: {event_id: token_id}
        Пример: {"[PAD]": 0, "[UNK]": 1, "E20020": 2, ...}
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Простой формат: {event_id: token_id}
        event_to_id = {}
        pad_id = 0
        unk_id = 1
        
        for event_id, token_id in data.items():
            if event_id == "[PAD]":
                pad_id = token_id
            elif event_id == "[UNK]":
                unk_id = token_id
            else:
                event_to_id[event_id] = token_id
        
        logger.info(
            f"✅ Загружен словарь: {len(event_to_id)} событий, "
            f"pad_id={pad_id}, unk_id={unk_id}"
        )

        return cls(
            event_to_id=event_to_id,
            pad_id=pad_id,
            unk_id=unk_id,
        )

    def cluster_id_to_token_id(self, cluster_id: Optional[str]) -> Tuple[str, int]:
        """
        cluster_id (из Drain3) -> (event_id, token_id).
        
        Логика:
        1. Если cluster_id=None -> возвращаем ("[UNK]", unk_id)
        2. Используем cluster_id напрямую как event_id
        3. Если event_id не найден в словаре -> возвращаем ("[UNK]", unk_id)
        """
        if cluster_id is None:
            return "[UNK]", self.unk_id

        # Используем cluster_id напрямую как event_id
        # ВАЖНО: во всём пайплайне события имеют вид "E123"
        # поэтому и здесь нужно добавить префикс "E"
        event_id = f"E{cluster_id}"

        # Получаем token_id для event_id
        token_id = self.event_to_id.get(event_id, self.unk_id)
        return event_id, token_id


# ---------------------------
# Thresholds
# ---------------------------

def load_nll_threshold(path: Path) -> float:
    """
    thresholds.json у тебя имеет вид:

    {
      "percentile": 99.5,
      "threshold_nll": 6.0849...,
      "count": 497063
    }

    Читаем threshold_nll и логируем метаданные.
    """
    if not path.exists():
        raise FileNotFoundError(f"Не найден thresholds.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "threshold_nll" not in data:
        raise ValueError(
            f"{path} не содержит обязательное поле 'threshold_nll'. "
            f"Пожалуйста, убедитесь, что файл thresholds.json содержит поле 'threshold_nll'."
        )

    thr = float(data["threshold_nll"])
    percentile = data.get("percentile")
    count = data.get("count")

    logger.info(
        "✅ Загрузили threshold_nll=%.6f (percentile=%s, count=%s) из %s",
        thr,
        percentile,
        count,
        path,
    )
    return thr


# ---------------------------
# Модель LogBERT
# ---------------------------

# Функции resolve_checkpoint_paths, get_device_from_config, load_model_from_checkpoint
# теперь импортируются из modules.model_utils


# ---------------------------
# Построение последовательностей и окон
# ---------------------------

def build_sequence_from_file(
    path: Path,
    template_miner: TemplateMiner,
    vocab: EventVocab,
    section_buckets: int,
    sect_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Обрабатывает файл логов через Drain3, повторяя логику encode из 09-drain_dataset.py.
    
    Важно:
    - Drain3 работает в read-only режиме (persistence_handler = None)
    - Если при обработке создается новый кластер - используется [UNK]
    - Разбивает логи на секции через parse_sections_from_lines
    - Создает mapping, sequences, chunks для сохранения
    
    Returns:
        Словарь с:
        - events: List[int] (token_id для LogBERT)
    - sections: List[int] (bucket индексы)
        - meta: List[dict] (метаданные по позиции)
        - mapping_records: List[dict] (для drain_mapping.jsonl)
        - event_seq: List[str] (для drain_sequences.jsonl, формат ["E1", "E2", ...])
        - chunks: List[dict] (для drain_chunks.jsonl)
    """
    pipeline_id, build_id = parse_filename(path)
    
    # Отслеживаем начальное количество кластеров для обнаружения новых
    initial_cluster_count = len(template_miner.drain.clusters)
    
    # Читаем все строки файла
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        return {
            "pipeline_id": pipeline_id,
            "build_id": build_id,
            "events": [],
            "sections": [],
            "meta": [],
            "mapping_records": [],
            "event_seq": [],
            "chunks": [],
        }
    
    # Разбиваем на секции (логика из encode_logs)
    sections_dict = parse_sections_from_lines(lines, sect_cfg)

    events: List[int] = []
    sections: List[int] = []
    meta: List[Dict[str, Any]] = []
    mapping_records: List[Dict[str, Any]] = []
    build_seq: List[str] = []  # Последовательность event_id в формате ["E1", "E2", ...]

    # Храним события по секциям для создания чанков
    section_events_dict: Dict[str, List[str]] = {}

    # Обрабатываем каждую секцию
    for section_name, section_lines in sections_dict.items():
        section_bucket = section_to_bucket(section_name, section_buckets)
        section_events: List[str] = []  # event_id в формате "E123"
        
        # Обрабатываем каждую строку секции
        for line_idx, line in enumerate(section_lines):
            # Запоминаем количество кластеров до обработки
            clusters_before = len(template_miner.drain.clusters)
            
            # Обрабатываем строку через Drain3
            result = template_miner.add_log_message(line)
            
            # Проверяем, не создался ли новый кластер
            clusters_after = len(template_miner.drain.clusters)
            is_new_cluster = clusters_after > clusters_before
            
            cluster_id = None
            if result and "cluster_id" in result:
                cluster_id = result["cluster_id"]

            # Если создался новый кластер - используем [UNK]
            if is_new_cluster:
                logger.warning(
                    "⚠️ Обнаружен новый паттерн в строке %d секции '%s' файла %s - используем [UNK]",
                    line_idx + 1, section_name, path.name
                )
                # Откатываем создание нового кластера невозможно, но мы просто используем [UNK]
                event_id = "[UNK]"
                token_id = vocab.unk_id
                cluster_id = None  # Помечаем как неизвестный
            else:
                # Нормальная обработка
                event_id, token_id = vocab.cluster_id_to_token_id(cluster_id)

            # Формируем event_id в формате "E123" для sequences и chunks
            if cluster_id is not None:
                eid = f"E{cluster_id}"
            else:
                eid = "[UNK]"
            
            section_events.append(eid)
            build_seq.append(eid)
            events.append(token_id)
            sections.append(section_bucket)
            
            # Метаданные для scored output
            meta.append(
                {
                    "pipeline_id": pipeline_id,
                    "build_id": build_id,
                    "line_no": line_idx + 1,  # 1-based в секции
                    "raw_line": line,
                    "section_name": section_name,
                    "cluster_id": cluster_id,
                    "event_id": event_id,
                }
            )
            
            # Запись для drain_mapping.jsonl
            cluster = template_miner.drain.id_to_cluster.get(cluster_id) if cluster_id else None
            template = cluster.get_template() if cluster else "<unknown>"
            mapping_records.append({
                "pipeline_id": pipeline_id,
                "build_id": build_id,
                "section_name": section_name,
                "event_id": eid,
                "template": template,
                "original_line": line,
                "file": str(path),
                "line_index": line_idx,
            })
        
        # Сохраняем события секции для создания чанков
        section_events_dict[section_name] = section_events

    return {
        "pipeline_id": pipeline_id,
        "build_id": build_id,
        "events": events,
        "sections": sections,
        "meta": meta,
        "mapping_records": mapping_records,
        "event_seq": build_seq,
        "section_events": section_events_dict,  # События по секциям для создания чанков
    }


def build_windows(
    events: List[int],
    sections: List[int],
    window_size: int,
    pad_id: int,
) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
    """
    Строит окна как при обучении next-event моделирования:
    
    Для последовательности длины L:
    - для target_pos в [window_size, L-1]:
        вход:  events[target_pos-window_size : target_pos]
        target: events[target_pos]
        sec_id: bucket секции target-события
    
    Возвращает:
      - input_ids_list:  List[List[int]] (окна длиной window_size)
      - sec_ids_list:    List[int] (один section_id на окно - секция target-события)
      - target_ids_list: List[int] (target token_id для каждого окна)
      - target_pos_list: List[int] (позиция target в исходной последовательности)
    """
    assert len(events) == len(sections)
    L = len(events)
    if L <= window_size:
        logger.warning(
            "Последовательность длины %d <= window_size=%d, окна не построены",
            L,
            window_size,
        )
        return [], [], [], []

    input_ids_list: List[List[int]] = []
    sec_ids_list: List[int] = []
    target_ids_list: List[int] = []
    target_pos_list: List[int] = []

    # Создаем окна только для валидных target позиций (как в prepare_windows)
    for target_pos in range(window_size, L):
        start = target_pos - window_size
        window_tokens = events[start:target_pos]  # Окно ДО target, длина = window_size
        
        if len(window_tokens) != window_size:
            # На всякий случай — не должно случаться
            continue

        input_ids_list.append(window_tokens)
        sec_ids_list.append(sections[target_pos])  # Секция target-события
        target_ids_list.append(events[target_pos])
        target_pos_list.append(target_pos)

    return input_ids_list, sec_ids_list, target_ids_list, target_pos_list


# ---------------------------
# Инференс логов
# ---------------------------

def infer_nll_per_position(
    model: torch.nn.Module,
    input_ids_list: List[List[int]],
    sec_ids_list: List[int],
    target_ids_list: List[int],
    target_pos_list: List[int],
    num_events: int,
    pad_id: int,
    batch_size: int,
) -> List[Optional[float]]:
    """
    Прогоняет все окна через модель, считает NLL для каждого target_pos.
    Если для одной позиции несколько окон → усредняем NLL.
    
    ВАЖНО: Логика должна совпадать с калибровкой (13-calibrate_threshold.py):
    - Используем nn.CrossEntropyLoss(reduction="none") для вычисления NLL
    - NLL вычисляется для каждого окна отдельно (как в калибровке)
    - Если позиция попадает в несколько окон как target — усредняем NLL
    """
    if not input_ids_list:
        return [None] * num_events

    device = next(model.parameters()).device

    # Инициализируем массивы для всех событий (длина = num_events)
    sums = [0.0 for _ in range(num_events)]
    counts = [0 for _ in range(num_events)]

    # Используем CrossEntropyLoss как в калибровке
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield i, lst[i:i + n]

    total_batches = math.ceil(len(input_ids_list) / batch_size)

    for start_idx, batch_input_ids in tqdm(
        chunks(input_ids_list, batch_size),
        total=total_batches,
        desc="Inference",
    ):
        batch_sec_ids = sec_ids_list[start_idx:start_idx + len(batch_input_ids)]
        batch_target_ids = target_ids_list[start_idx:start_idx + len(batch_input_ids)]
        batch_target_pos = target_pos_list[start_idx:start_idx + len(batch_input_ids)]

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)  # (B, S)
        sec_ids = torch.tensor(batch_sec_ids, dtype=torch.long, device=device)  # (B,)
        target_ids_tensor = torch.tensor(batch_target_ids, dtype=torch.long, device=device)  # (B,)

        with torch.no_grad():
            logits = model(input_ids, sec_ids)  # (B, V)
            # Вычисляем NLL для каждого окна (как в калибровке)
            loss = ce(logits, target_ids_tensor)  # (B,)

        # Для каждого окна сохраняем NLL для его target-позиции
        for b, global_pos in enumerate(batch_target_pos):
            nll = loss[b].item()
            sums[global_pos] += nll
            counts[global_pos] += 1

    # Усредняем NLL для позиций, которые были target в нескольких окнах
    nll_per_pos: List[Optional[float]] = []
    for s, c in zip(sums, counts):
        if c == 0:
            nll_per_pos.append(None)
        else:
            nll_per_pos.append(s / c)

    return nll_per_pos


# ---------------------------
# Основной цикл по файлам
# ---------------------------

def process_single_file(
    file_path: Path,
    template_miner: TemplateMiner,
    vocab: EventVocab,
    model: torch.nn.Module,
    nll_threshold: float,
    config: Config,
    scored_dir: Path,
    drain_dir: Path,
) -> None:
    """
    Обрабатывает один файл логов и сохраняет результаты в scored_dir.
    Также сохраняет drain_mapping.jsonl, drain_sequences.jsonl, drain_chunks.jsonl в drain_dir.
    
    Args:
        file_path: Путь к входному файлу *-logs_content.lst
        template_miner: Drain3 miner для парсинга логов (read-only режим)
        vocab: Словарь событий
        model: Обученная модель LogBERT
        nll_threshold: Порог NLL для детектирования аномалий
        config: Конфигурация из config.yaml (объект Config)
        scored_dir: Директория для сохранения результатов (inference.scored_dir)
        drain_dir: Директория для сохранения drain файлов (inference.drain_dir)
    """
    windows_cfg = config.get("dataset.windows")
    if not windows_cfg:
        raise ValueError(
            "Секция 'dataset.windows' не найдена в config.yaml. "
            "Пожалуйста, добавьте секцию dataset.windows с полями size и batch_size."
        )
    
    window_size = windows_cfg.get("size")
    if window_size is None:
        raise ValueError(
            "Поле 'dataset.windows.size' не указано в config.yaml. "
            "Пожалуйста, укажите размер окна для обработки последовательностей."
        )
    
    batch_size = windows_cfg.get("batch_size")
    if batch_size is None:
        raise ValueError(
            "Поле 'dataset.windows.batch_size' не указано в config.yaml. "
            "Пожалуйста, укажите размер батча для инференса."
        )
    
    # Получаем настройки секций и чанков
    dataset_cfg = config.get("dataset", {})
    sect_cfg = dataset_cfg.get("section_markers", {})
    min_seq_len = int(dataset_cfg.get("min_seq_len", 5))
    max_chunk_len = int(dataset_cfg.get("max_chunk_len", 512))
    chunk_stride = int(dataset_cfg.get("chunk_stride", 256))

    logger.info("▶ Обработка файла %s", file_path)

    seq = build_sequence_from_file(
        file_path,
        template_miner=template_miner,
        vocab=vocab,
        section_buckets=int(config.get("model.section_buckets")),
        sect_cfg=sect_cfg,
    )

    events = seq["events"]
    sections = seq["sections"]
    meta = seq["meta"]
    mapping_records = seq["mapping_records"]
    event_seq = seq["event_seq"]
    section_events = seq["section_events"]
    pipeline_id = seq["pipeline_id"]
    build_id = seq["build_id"]

    if not events:
        logger.warning("Файл %s пустой после предобработки, пропускаем", file_path)
        return

    # Создаем чанки из событий по секциям (логика из encode_logs)
    chunks_records = []
    for section_name, section_evts in section_events.items():
        for ch in chunk_sequence(section_evts, max_chunk_len, chunk_stride, min_seq_len):
            chunks_records.append({
                "pipeline_id": pipeline_id,
                "build_id": build_id,
                "section_name": section_name,
                "event_seq": ch,
                "status": "error",  # Инференс обрабатывает ошибочные логи
            })

    # Сохраняем drain файлы в inference.drain_dir
    ensure_dir(drain_dir)
    
    # 1. drain_mapping.jsonl (append mode)
    mapping_file = drain_dir / "drain_mapping.jsonl"
    with open(mapping_file, "a", encoding="utf-8") as f:
        for rec in mapping_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("💾 Сохранено %d записей в drain_mapping.jsonl", len(mapping_records))
    
    # 2. drain_sequences.jsonl (append mode)
    sequences_file = drain_dir / "drain_sequences.jsonl"
    sequence_record = {
        "pipeline_id": pipeline_id,
        "build_id": build_id,
        "event_seq": event_seq,
        "status": "error",  # Инференс обрабатывает ошибочные логи
    }
    with open(sequences_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(sequence_record, ensure_ascii=False) + "\n")
    logger.info("💾 Сохранена последовательность в drain_sequences.jsonl")
    
    # 3. drain_chunks.jsonl (append mode)
    chunks_file = drain_dir / "drain_chunks.jsonl"
    with open(chunks_file, "a", encoding="utf-8") as f:
        for rec in chunks_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("💾 Сохранено %d чанков в drain_chunks.jsonl", len(chunks_records))

    # Инференс через модель
    input_ids_list, sec_ids_list, target_ids_list, target_pos_list = build_windows(
        events=events,
        sections=sections,
        window_size=window_size,
        pad_id=vocab.pad_id,
    )

    num_events = len(events)
    nll_per_pos = infer_nll_per_position(
        model=model,
        input_ids_list=input_ids_list,
        sec_ids_list=sec_ids_list,
        target_ids_list=target_ids_list,
        target_pos_list=target_pos_list,
        num_events=num_events,
        pad_id=vocab.pad_id,
        batch_size=batch_size,
    )

    # Создаем директорию для выходных файлов если нужно
    ensure_dir(scored_dir)
    
    # Выходной файл: заменяем "logs_content" на "logs_scored" и сохраняем в scored_dir
    out_name = file_path.stem.replace("logs_content", "logs_scored") + ".jsonl"
    out_path = scored_dir / out_name
    logger.info("💾 Сохраняем результаты инференса в %s", out_path)

    # Статистика для логирования
    total_lines = len(meta)
    anomalies_count = 0
    unk_count = 0
    unk_anomalies_count = 0
    nll_anomalies_count = 0
    max_nll = None
    min_nll = None
    nll_sum = 0.0
    nll_count = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for pos, m in enumerate(meta):
            nll = nll_per_pos[pos] if pos < len(nll_per_pos) else None
            
            # Аномалия если:
            # 1. NLL превышает порог, ИЛИ
            # 2. Это новый паттерн (cluster_id is None) - не было в обучающих данных
            is_nll_anomaly = bool(nll is not None and nll > nll_threshold)
            is_unk = m.get("cluster_id") is None
            is_anomaly = is_nll_anomaly or is_unk

            if is_anomaly:
                anomalies_count += 1
            if is_unk:
                unk_count += 1
                if is_anomaly:
                    unk_anomalies_count += 1
            if is_nll_anomaly:
                nll_anomalies_count += 1
            
            # Статистика по NLL
            if nll is not None:
                if max_nll is None or nll > max_nll:
                    max_nll = nll
                if min_nll is None or nll < min_nll:
                    min_nll = nll
                nll_sum += nll
                nll_count += 1

            record = {
                **m,
                "position": pos,
                "nll": nll,
                "is_anomaly": is_anomaly,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Сохраняем статистику инференса
    avg_nll = nll_sum / nll_count if nll_count > 0 else None
    inference_stats = {
        "pipeline_id": pipeline_id,
        "build_id": build_id,
        "file_path": str(file_path),
        "output_file": str(out_path),
        "total_lines": total_lines,
        "anomalies_count": anomalies_count,
        "anomalies_percent": (anomalies_count / total_lines * 100) if total_lines > 0 else 0.0,
        "unk_count": unk_count,
        "unk_anomalies_count": unk_anomalies_count,
        "nll_anomalies_count": nll_anomalies_count,
        "nll_threshold": nll_threshold,
        "max_nll": max_nll,
        "min_nll": min_nll,
        "avg_nll": avg_nll,
        "nll_count": nll_count,
    }
    
    # Сохраняем статистику в inference_summary.jsonl
    summary_file = scored_dir / "inference_summary.jsonl"
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(inference_stats, ensure_ascii=False) + "\n")

    logger.info("✅ Готово: %s", out_path)
    logger.info(
        "📊 Статистика: всего строк=%d, аномалий=%d (%.1f%%), "
        "новых паттернов [UNK]=%d (все помечены как аномалии), "
        "аномалий по NLL=%d, avg_nll=%.4f",
        total_lines, anomalies_count, 
        (anomalies_count / total_lines * 100) if total_lines > 0 else 0.0,
        unk_count, nll_anomalies_count, avg_nll if avg_nll else 0.0
    )


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    """
    Основная функция для запуска инференса LogBERT.
    
    Использует настройки из config.yaml:
    - gpu.device: устройство для вычислений (cuda/mps/cpu)
    - dataset.drain: директория с неизменяемым Drain3 state (drain_state.json, drain_templates.json)
    - inference.drain_dir: директория для сохранения новых данных инференса (маппинг, последовательности)
    - inference.scored_dir: директория для сохранения результатов (*-logs_scored.jsonl)
    - dataset.prepared: директория с входными файлами *-logs_content.lst
    - dataset.vocab: директория со словарем event_vocab.json
    - model.checkpoints: путь к чекпоинту модели
    """
    parser = argparse.ArgumentParser(description="Инференс LogBERT на подготовленных логах")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к config.yaml",
    )
    args = parser.parse_args()

    # Загружаем конфигурацию из modules.config
    # Если указан другой путь, создаем новый экземпляр Config
    if args.config != "config.yaml":
        from modules.config import Config
        config = Config(args.config)
    else:
        config = get_config()

    # --- Пути из config.yaml ---
    dataset_cfg = config.get_dataset_config()
    prepared_dir_str = dataset_cfg.get("prepared")
    if not prepared_dir_str:
        raise ValueError(
            "Поле 'dataset.prepared' не указано в config.yaml. "
            "Пожалуйста, укажите путь к директории с подготовленными файлами."
        )
    prepared_dir = Path(prepared_dir_str)
    
    # Неизменяемый state Drain3 загружается из dataset.drain
    drain_state_dir_str = dataset_cfg.get("drain")
    if not drain_state_dir_str:
        raise ValueError(
            "Поле 'dataset.drain' не указано в config.yaml. "
            "Пожалуйста, укажите путь к директории с неизменяемым Drain3 state (drain_state.json, drain_templates.json)."
        )
    drain_state_dir = Path(drain_state_dir_str)
    
    # Новые данные инференса сохраняются в inference.drain_dir
    inference_cfg = config.get("inference")
    if not inference_cfg:
        raise ValueError(
            "Секция 'inference' не найдена в config.yaml. "
            "Пожалуйста, добавьте секцию inference с полями drain_dir и scored_dir."
        )
    
    inference_drain_dir_str = inference_cfg.get("drain_dir")
    if not inference_drain_dir_str:
        raise ValueError(
            "Поле 'inference.drain_dir' не указано в config.yaml. "
            "Пожалуйста, укажите путь к директории для сохранения новых данных инференса."
        )
    
    scored_dir_str = inference_cfg.get("scored_dir")
    if not scored_dir_str:
        raise ValueError(
            "Поле 'inference.scored_dir' не указано в config.yaml. "
            "Пожалуйста, укажите путь к директории для сохранения результатов."
        )
    
    inference_drain_dir = Path(inference_drain_dir_str)
    scored_dir = Path(scored_dir_str)
    
    vocab_dir_str = dataset_cfg.get("vocab")
    if not vocab_dir_str:
        raise ValueError(
            "Поле 'dataset.vocab' не указано в config.yaml. "
            "Пожалуйста, укажите путь к директории со словарем."
        )
    vocab_dir = Path(vocab_dir_str)

    model_ckpt_path, thresholds_path = resolve_checkpoint_paths(config)

    logger.info("📁 Конфигурация путей:")
    logger.info("   - Входные файлы: %s", prepared_dir)
    logger.info("   - Drain3 state (read-only): %s", drain_state_dir)
    logger.info("   - Drain3 новые данные: %s", inference_drain_dir)
    logger.info("   - Результаты: %s", scored_dir)
    logger.info("   - Словарь: %s", vocab_dir)
    logger.info("   - Модель: %s", model_ckpt_path)
    logger.info("   - Пороги: %s", thresholds_path)

    # Загрузка компонентов
    # Загружаем state из dataset.drain (read-only), новые данные сохраняем в inference.drain_dir
    template_miner = load_drain(drain_state_dir, inference_drain_dir)

    event_vocab_path = vocab_dir / "event_vocab.json"
    vocab = EventVocab.from_json(event_vocab_path)

    nll_threshold = load_nll_threshold(thresholds_path)

    vocab_size = max(vocab.event_to_id.values()) + 1
    model = load_model_from_checkpoint(config, model_ckpt_path, vocab_size=vocab_size)

    # Проходим по всем *-logs_content.lst
    lst_files = sorted(prepared_dir.glob("*-logs_content.lst"))
    if not lst_files:
        logger.warning("В %s не найдено файлов *-logs_content.lst", prepared_dir)
        return

    logger.info("📋 Найдено файлов для обработки: %d", len(lst_files))

    for file_path in lst_files:
        process_single_file(
            file_path=file_path,
            template_miner=template_miner,
            vocab=vocab,
            model=model,
            nll_threshold=nll_threshold,
            config=config,
            scored_dir=scored_dir,
            drain_dir=inference_drain_dir,
        )
    
    # Подсчитываем итоговую статистику по всем файлам
    summary_file = scored_dir / "inference_summary.jsonl"
    if summary_file.exists():
        total_files = 0
        total_lines = 0
        total_anomalies = 0
        total_unk = 0
        total_nll_anomalies = 0
        all_max_nll = None
        all_min_nll = None
        nll_sum_all = 0.0
        nll_count_all = 0
        
        with open(summary_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats = json.loads(line)
                total_files += 1
                total_lines += stats.get("total_lines", 0)
                total_anomalies += stats.get("anomalies_count", 0)
                total_unk += stats.get("unk_count", 0)
                total_nll_anomalies += stats.get("nll_anomalies_count", 0)
                
                if stats.get("max_nll") is not None:
                    if all_max_nll is None or stats["max_nll"] > all_max_nll:
                        all_max_nll = stats["max_nll"]
                if stats.get("min_nll") is not None:
                    if all_min_nll is None or stats["min_nll"] < all_min_nll:
                        all_min_nll = stats["min_nll"]
                if stats.get("avg_nll") is not None and stats.get("nll_count", 0) > 0:
                    nll_sum_all += stats["avg_nll"] * stats["nll_count"]
                    nll_count_all += stats["nll_count"]
        
        avg_nll_all = nll_sum_all / nll_count_all if nll_count_all > 0 else None
        
        logger.info("=" * 80)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА ИНФЕРЕНСА")
        logger.info("=" * 80)
        logger.info("   Обработано файлов: %d", total_files)
        logger.info("   Всего строк: %d", total_lines)
        logger.info("   Всего аномалий: %d (%.2f%%)", total_anomalies, 
                   (total_anomalies / total_lines * 100) if total_lines > 0 else 0.0)
        logger.info("   Новых паттернов [UNK]: %d (все помечены как аномалии)", total_unk)
        logger.info("   Аномалий по NLL: %d", total_nll_anomalies)
        logger.info("   NLL статистика:")
        logger.info("      - Средний NLL: %.4f", avg_nll_all if avg_nll_all else 0.0)
        logger.info("      - Минимальный NLL: %.4f", all_min_nll if all_min_nll else 0.0)
        logger.info("      - Максимальный NLL: %.4f", all_max_nll if all_max_nll else 0.0)
        logger.info("      - Порог NLL: %.4f", nll_threshold)
        logger.info("=" * 80)
        logger.info("✅ Инференс завершен. Результаты сохранены в %s", scored_dir)
        logger.info("📄 Детальная статистика по файлам: %s", summary_file)
    else:
        logger.info("✅ Инференс завершен. Результаты сохранены в %s", scored_dir)


if __name__ == "__main__":
    main()
