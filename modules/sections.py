# -*- coding: utf-8 -*-
"""
modules/sections.py — утилиты для работы с секциями логов и чанкованием последовательностей.

Функции:
  - parse_sections_from_lines: разбивает логи на секции по маркерам (Azure DevOps, TFS)
  - chunk_sequence: разбивает последовательность событий на чанки

Используется в:
  - 09-drain_dataset.py (--encode, --append)
  - 14-run_inference.py (обработка логов для инференса)
"""
import re
from typing import Dict, List, Any


def parse_sections_from_lines(lines: List[str], sect_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Разбивает логи на секции по маркерам (Azure DevOps, TFS).
    
    Поддерживает:
    - Azure DevOps маркеры: ##[section]Starting: и ##[section]Finishing:
    - TFS маркеры: <TFS_Section> (настраивается через regex)
    
    Args:
        lines: Список строк лога
        sect_cfg: Конфигурация секций из config.yaml (dataset.section_markers)
            - use_azure_sections: bool (по умолчанию True)
            - use_tfs_sections: bool (по умолчанию True)
            - tfs_regex: str (по умолчанию r"<TFS_Section>\s*(.+)")
    
    Returns:
        Словарь {section_name: [строки секции]}
        Если секции не найдены, возвращает {"full": lines}
    """
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


def chunk_sequence(seq: List[str], max_len: int, stride: int, min_len: int) -> List[List[str]]:
    """
    Разбивает последовательность событий на чанки.
    
    Создает перекрывающиеся чанки с заданным шагом (stride).
    Чанки короче min_len отбрасываются.
    
    Args:
        seq: Последовательность событий (например, ["E1", "E2", "E3", ...])
        max_len: Максимальная длина чанка
        stride: Шаг для создания перекрывающихся чанков
        min_len: Минимальная длина чанка (чанки короче min_len отбрасываются)
    
    Returns:
        Список чанков (каждый чанк - это List[str])
        Если последовательность пустая, возвращает []
        Если последовательность короче max_len, возвращает [seq] если len(seq) >= min_len, иначе []
    
    Пример:
        >>> chunk_sequence(["E1", "E2", "E3", "E4", "E5"], max_len=3, stride=2, min_len=2)
        [["E1", "E2", "E3"], ["E3", "E4", "E5"]]
    """
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

