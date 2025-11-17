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
    id_of=lambda e:vocab.get(e, UNK_ID)
    X,Y,S,meta=[],[],[],[]
    for p in jsonl_paths:
        for rec in load_jsonl(p):
            if accept_status and rec.get('status')!=accept_status: continue
            seq=rle_cap(rec.get('event_seq',[]),rle_cap_n)
            if len(seq)<=window_size: continue
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
    if not X: raise RuntimeError('No windows produced')
    X_tensor: torch.Tensor = torch.tensor(X,dtype=torch.long)
    Y_tensor: torch.Tensor = torch.tensor(Y,dtype=torch.long)
    S_tensor: torch.Tensor = torch.tensor(S,dtype=torch.long)
    ensure_dir(out_pt.parent)
    torch.save({'input_ids':X_tensor,'target_ids':Y_tensor,'sec_ids':S_tensor},out_pt)
    ensure_dir(out_meta.parent)
    write_jsonl(out_meta,[asdict(m) for m in meta])
    log.info(f"[Prepare] Saved {len(X_tensor)} windows -> {out_pt}")
