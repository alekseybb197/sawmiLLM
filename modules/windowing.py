# -*- coding: utf-8 -*-
"""
modules/windowing.py — подготовка оконных выборок для LogBERT.

Функции:
  - rle_cap: ограничивает длину последовательных повторов событий.
  - section_to_bucket: хэширование названия секции в ограниченный диапазон.
  - prepare_windows: читает JSONL с event_seq, режет на окна, сохраняет .pt и мета-информацию.

Используется как для success, так и для error логов.

Выход:
  prepared/<split>_<tag>.pt — dict(input_ids, target_ids, sec_ids)
  prepared/<split>_<tag>_meta.jsonl — WindowMeta для обратного маппинга.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Union
import hashlib
import json
import torch

from modules.logger import get_logger
from modules.utils import ensure_dir, load_jsonl, write_jsonl

log = get_logger(__name__)

PAD_ID = 0
UNK_ID = 1

def section_to_bucket(section_name: str, buckets: int = 2048) -> int:
    h = hashlib.md5(section_name.encode('utf-8')).hexdigest()
    return int(h, 16) % buckets

def rle_cap(seq: List[str], cap: int) -> List[str]:
    if cap<=0 or not seq: return list(seq)
    out=[]; last=None; c=0
    for s in seq:
        if s==last: c+=1
        else: last=s; c=1
        if c<=cap: out.append(s)
    return out

@dataclass
class WindowMeta:
    pipeline_id: str
    build_id: str
    section_name: str
    section_bucket: int
    target_pos_in_section: int
    window_idx_in_section: int

def prepare_windows(jsonl_paths: List[Path], vocab: Dict[str,int], out_pt: Path, out_meta: Path,
                    window_size:int=32, stride:int=1, rle_cap_n:int=5, section_buckets:int=2048,
                    accept_status:Optional[str]=None) -> None:
    """
    Подготавливает оконные выборки для LogBERT.
    
    ВАЖНО: Все события должны присутствовать в словаре. Записи с отсутствующими событиями
    пропускаются, так как [UNK] используется только для аномалий и не должен появляться
    в обучающих/валидационных данных.
    """
    # Получаем unk_id из словаря (должен быть 1)
    unk_id = vocab.get("[UNK]", UNK_ID)
    
    # Функция для получения token_id - БЕЗ fallback на UNK
    def id_of(e: str) -> int:
        if e not in vocab:
            raise ValueError(f"Event '{e}' not found in vocabulary. This should not happen in train/val data.")
        return vocab[e]
    
    X,Y,S,meta=[],[],[],[]
    skipped_records = 0
    skipped_reasons: Dict[str, int] = {}
    missing_events: set[str] = set()
    
    for p in jsonl_paths:
        for rec in load_jsonl(p):
            if accept_status and rec.get('status')!=accept_status: 
                continue
            
            seq=rle_cap(rec.get('event_seq',[]),rle_cap_n)
            if len(seq)<=window_size: 
                skipped_records += 1
                skipped_reasons["too_short"] = skipped_reasons.get("too_short", 0) + 1
                continue
            
            # Проверяем наличие всех событий в словаре ПЕРЕД обработкой
            missing = [e for e in seq if e not in vocab]
            if missing:
                skipped_records += 1
                skipped_reasons["missing_events"] = skipped_reasons.get("missing_events", 0) + 1
                missing_events.update(missing)
                log.warning(
                    f"[Prepare] Skipping record pipeline_id={rec.get('pipeline_id')} "
                    f"build_id={rec.get('build_id')}: missing events in vocab: {missing[:5]}"
                    f"{'...' if len(missing) > 5 else ''}"
                )
                continue
            
            # Все события присутствуют в словаре - обрабатываем
            try:
                sid=section_to_bucket(rec.get('section_name',''),section_buckets)
                ids=[id_of(e) for e in seq]
                w=0
                for i in range(0,len(ids)-window_size,stride):
                    x=ids[i:i+window_size]; y=ids[i+window_size]
                    X.append(x); Y.append(y); S.append(sid)
                    meta.append(WindowMeta(
                        pipeline_id=str(rec.get('pipeline_id')),
                        build_id=str(rec.get('build_id')),
                        section_name=str(rec.get('section_name')),
                        section_bucket=sid,
                        target_pos_in_section=i+window_size,
                        window_idx_in_section=w
                    )); w+=1
            except ValueError as e:
                # На всякий случай - если все же попалось отсутствующее событие
                skipped_records += 1
                skipped_reasons["conversion_error"] = skipped_reasons.get("conversion_error", 0) + 1
                log.warning(f"[Prepare] Skipping record due to error: {e}")
                continue
    
    if not X: 
        raise RuntimeError(
            f'No windows produced. Skipped {skipped_records} records. '
            f'Reasons: {skipped_reasons}. Missing events: {sorted(missing_events)[:10]}'
        )
    
    # Проверяем, что в тензорах нет UNK
    X_tensor: torch.Tensor = torch.tensor(X,dtype=torch.long)
    Y_tensor: torch.Tensor = torch.tensor(Y,dtype=torch.long)
    S_tensor: torch.Tensor = torch.tensor(S,dtype=torch.long)
    
    # Валидация: проверяем отсутствие UNK в тензорах
    if (X_tensor == unk_id).any():
        unk_count_input = (X_tensor == unk_id).sum().item()
        raise RuntimeError(
            f"CRITICAL: Found {unk_count_input} [UNK] tokens in input_ids! "
            f"This should not happen in train/val data. "
            f"Missing events in vocab: {sorted(missing_events)[:20]}"
        )
    
    if (Y_tensor == unk_id).any():
        unk_count_target = (Y_tensor == unk_id).sum().item()
        raise RuntimeError(
            f"CRITICAL: Found {unk_count_target} [UNK] tokens in target_ids! "
            f"This should not happen in train/val data. "
            f"Missing events in vocab: {sorted(missing_events)[:20]}"
        )
    
    ensure_dir(out_pt.parent)
    torch.save({'input_ids':X_tensor,'target_ids':Y_tensor,'sec_ids':S_tensor},out_pt)
    ensure_dir(out_meta.parent)
    write_jsonl(out_meta,[asdict(m) for m in meta])
    
    log.info(f"[Prepare] Saved {len(X_tensor)} windows -> {out_pt}")
    if skipped_records > 0:
        log.warning(
            f"[Prepare] Skipped {skipped_records} records. Reasons: {skipped_reasons}. "
            f"Unique missing events: {len(missing_events)}"
        )
        if missing_events:
            log.warning(f"[Prepare] Sample of missing events: {sorted(missing_events)[:10]}")
