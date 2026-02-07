# -*- coding: utf-8 -*-
"""
modules/vocab.py — построение словаря событий Drain для LogBERT.

Функции:
  - build_event_vocab: из success JSONL и drain_templates.json собирает словарь событий (E###)
    и сохраняет event_vocab.json + event_vocab_stats.json

Особенности:
  - Использует только status == "success".
  - Специальные токены: [PAD]=0, [UNK]=1.
  - Детерминированная сортировка событий (по частоте, затем лексикографически).
  - Поддержка min_support и exclude_patterns для фильтрации.
  - Отчёт со статистикой частот, фильтрацией и отладочной информацией.

Зависимости: только стандартная библиотека + modules.logger / modules.utils.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Any, Iterable

from modules.logger import get_logger
from modules.utils import ensure_dir

log = get_logger(__name__)

PAD_ID = 0
UNK_ID = 1
SPECIAL_TOKENS = {"[PAD]": PAD_ID, "[UNK]": UNK_ID}
EVENT_RE = re.compile(r"^E\d+$")


@dataclass
class VocabStats:
    total_files: int
    total_records: int
    success_records: int
    unique_events: int
    used_templates: int
    filtered_by_pattern: int
    filtered_by_support: int
    event_freq_top10: List[Tuple[str, int]]
    malformed_records: int
    non_event_tokens: int
    min_support: int
    exclude_patterns: List[str]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                log.warning(f"{path}:{ln}: JSON parse error: {e}")
                continue


def _gather_jsonl(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.jsonl")))
        elif p.is_file():
            out.append(p)
        else:
            log.warning(f"Path not found: {p}")
    return out


def build_event_vocab(
    inputs: List[Path],
    outdir: Path,
    drain_templates_path: Path | None = None,
    min_support: int = 1,
    exclude_patterns: List[str] | None = None,  # Игнорируется, оставлено для совместимости
) -> Tuple[Path, Path]:
    """Строит словарь событий из success-записей + drain_templates.json.

    ВАЖНО: 
    - Частоты считаются ТОЛЬКО из success-записей (status == "success")
    - События, которые встречаются только в error-записях, НЕ попадут в словарь
    - Если событие есть в drain_templates.json, но не встречалось в success-записях,
      оно будет включено в словарь только если min_support = 0 (не рекомендуется)
    
    :param inputs: список путей к JSONL файлам или директориям
    :param outdir: папка для сохранения event_vocab.json и event_vocab_stats.json
    :param drain_templates_path: путь к drain_templates.json (по умолчанию ./drain_templates.json)
    :param min_support: минимальная частота включения события (из success-записей)
    :param exclude_patterns: ИГНОРИРУЕТСЯ (удалено из логики)
    :return: (path_to_vocab_json, path_to_stats_json)
    """
    ensure_dir(outdir)
    jsonl_files = _gather_jsonl(inputs)
    if not jsonl_files:
        raise FileNotFoundError("No JSONL files found in provided inputs")

    log.info(f"[Vocab] Scanning {len(jsonl_files)} files…")

    # === 1. Сбор частот из JSONL (ТОЛЬКО success-записи) ===
    freq: Counter[str] = Counter()
    total_records = 0
    success_records = 0
    error_records = 0
    malformed = 0
    non_event_tokens = 0

    for fp in jsonl_files:
        for rec in _iter_jsonl(fp):
            total_records += 1
            status = rec.get("status", "unknown")
            if status != "success":
                if status == "error":
                    error_records += 1
                continue
            success_records += 1
            ev = rec.get("event_seq")
            if not isinstance(ev, list):
                malformed += 1
                continue
            for tok in ev:
                if not isinstance(tok, str) or not EVENT_RE.match(tok):
                    non_event_tokens += 1
                    continue
                freq[tok] += 1

    log.info(
        f"[Vocab] Records: total={total_records}, success={success_records}, "
        f"error={error_records}, malformed={malformed}"
    )

    # === 2. Загрузка шаблонов Drain ===
    drain_templates_path = drain_templates_path or Path("drain_templates.json")
    if not drain_templates_path.exists():
        raise FileNotFoundError(f"Drain templates not found: {drain_templates_path}")
    with drain_templates_path.open("r", encoding="utf-8") as f:
        templates: Dict[str, Any] = json.load(f)

    # === 3. Фильтрация шаблонов (только по min_support) ===
    filtered_by_support = 0
    used_templates = 0
    filtered_templates: Dict[str, Dict[str, Any]] = {}

    for eid, tpl in templates.items():
        tpl_str = tpl["template"] if isinstance(tpl, dict) and "template" in tpl else str(tpl)
        freq_val = freq.get(eid, 0)
        
        # ВАЖНО: если событие не встречалось в success-записях, freq_val = 0
        # При min_support = 1 такие события будут отфильтрованы
        if freq_val < min_support:
            filtered_by_support += 1
            continue
        
        used_templates += 1
        filtered_templates[eid] = {"template": tpl_str, "freq": freq_val}

    # === 5. Формирование словаря ===
    sorted_items = sorted(filtered_templates.items(), key=lambda x: (-x[1]["freq"], x[0]))
    vocab = {**SPECIAL_TOKENS}
    for i, (eid, _) in enumerate(sorted_items, start=len(SPECIAL_TOKENS)):
        vocab[eid] = i

    # === 6. Сохранение ===
    vocab_path = outdir / "event_vocab.json"
    stats_path = outdir / "event_vocab_stats.json"

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    stats = VocabStats(
        total_files=len(jsonl_files),
        total_records=total_records,
        success_records=success_records,
        unique_events=len(freq),
        used_templates=used_templates,
        filtered_by_pattern=0,  # Больше не используется
        filtered_by_support=filtered_by_support,
        event_freq_top10=top10,
        malformed_records=malformed,
        non_event_tokens=non_event_tokens,
        min_support=min_support,
        exclude_patterns=[],  # Больше не используется
    )
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)

    log.info(
        f"[Vocab] Built {len(vocab)} tokens (incl specials). "
        f"Used templates={used_templates}/{len(templates)}. "
        f"Min_support={min_support}. Saved: {vocab_path}"
    )
    
    if filtered_by_support > 0:
        log.warning(
            f"[Vocab] Filtered {filtered_by_support} events by min_support={min_support}. "
            f"These events did not appear in success records (or appeared < {min_support} times). "
            f"See {stats_path}"
        )
        log.warning(
            f"[Vocab] ⚠️  ВНИМАНИЕ: Отфильтрованные события НЕ будут включены в словарь. "
            f"Если они присутствуют в train/val данных, записи с ними будут пропущены при создании тензоров. "
            f"[UNK] используется только для аномалий и не должен появляться в обучающих данных."
        )
        log.warning(
            f"[Vocab] 💡 Подсказка: Если нужно включить все события из drain_templates.json, "
            f"установите min_support=0 (не рекомендуется, так как может включить события, "
            f"которые не встречались в success-записях)."
        )
    if malformed or non_event_tokens:
        log.warning(
            f"[Vocab] Malformed={malformed}, non_event_tokens={non_event_tokens}. See {stats_path}"
        )
    return vocab_path, stats_path
