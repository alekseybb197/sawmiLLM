# -*- coding: utf-8 -*-
"""
modules/model.py — архитектура unified LogBERT:
  - PositionalEncoding (sin/cos)
  - UnifiedLogBERT (Embedding токенов + секций, TransformerEncoder, LM head)
Оптимизировано для RTX 4080 (поддержка AMP на уровне скрипта обучения).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import cast

PAD_ID = 0

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        pe_buffer = cast(torch.Tensor, self.pe)
        return x + pe_buffer[:, :L, :]

class UnifiedLogBERT(nn.Module):                   # Определяем класс нейросети, наследуемый от базового torch.nn.Module
    def __init__(                                   # Конструктор модели — принимает размеры и гиперпараметры
        self,
        vocab_size: int,                            # Размер словаря событий (кол-во уникальных токенов E###)
        section_buckets: int = 2048,                # Кол-во возможных "секции/шагов" пайплайна — для отдельной embedding-таблицы
        d_model: int = 512,                         # Размерность скрытого представления (векторов токенов)
        n_heads: int = 8,                           # Количество голов в multi-head attention
        n_layers: int = 6,                          # Глубина трансформера — сколько слоёв encoder’а
        dim_ff: int = 2048,                         # Размер внутреннего слоя в feed-forward блоке
        dropout: float = 0.1,                       # Вероятность дропаута для регуляризации
        max_len: int = 256,                         # Максимальная длина входной последовательности (число событий)
    ):
        super().__init__()                          # Инициализируем базовый класс nn.Module

        # --- Эмбеддинги токенов ---
        self.token_emb = nn.Embedding(              # Таблица эмбеддингов для событий E1, E2, ..., En
            vocab_size,                             # количество строк = число событий в словаре
            d_model,                                # размер векторного представления каждого события
            padding_idx=PAD_ID                      # специальный индекс для паддинга (градиенты не обновляются)
        )

        # --- Эмбеддинги секций/пайплайнов ---
        self.section_emb = nn.Embedding(            # Дополнительные эмбеддинги для "bucket’ов" — групп логов (например, стадия сборки)
            section_buckets,                        # количество возможных секций (в train задаётся по числу уникальных section_name)
            d_model                                 # размерность совпадает с d_model, чтобы можно было суммировать
        )

        # --- Позиционное кодирование ---
        self.pos = PositionalEncoding(              # Модуль добавляет позиционные смещения (sin/cos или learnable)
            d_model,                                # размер векторного пространства
            max_len=max_len                         # длина, на которую заранее генерируются позиции
        )

        # --- Базовый слой Transformer Encoder ---
        layer = nn.TransformerEncoderLayer(         # Один слой энкодера (Attention + FeedForward)
            d_model=d_model,                        # размер входных/выходных векторов
            nhead=n_heads,                          # количество attention-голов
            dim_feedforward=dim_ff,                 # размер скрытого слоя внутри FFN
            dropout=dropout,                        # дропаут на attention и FFN
            activation='gelu',                      # активация в FFN (обычно GELU для BERT-подобных моделей)
            batch_first=True                        # формат входа: (batch, seq_len, hidden)
        )

        # --- Стек энкодеров ---
        self.encoder = nn.TransformerEncoder(       # Собираем несколько таких слоёв в стек
            layer,                                  # шаблон слоя, определённый выше
            num_layers=n_layers                     # количество слоёв в глубину (6 по умолчанию)
        )

        # --- Нормализация выходных признаков ---
        self.norm = nn.LayerNorm(d_model)           # LayerNorm стабилизирует распределение активаций после энкодера

        # --- Голова предсказания токена (Language Modeling Head) ---
        self.lm_head = nn.Linear(                   # Линейный слой: скрытое представление → вероятности по словарю событий
            d_model,                                # входная размерность (из трансформера)
            vocab_size                              # выходная размерность = число событий (E###)
        )


    def forward(self, input_ids: torch.Tensor, sec_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)             # (B,L,D)
        s = self.section_emb(sec_ids).unsqueeze(1) # (B,1,D)
        x = x + s
        x = self.pos(x)
        h = self.encoder(x)                       # (B,L,D)
        h = self.norm(h[:, -1, :])                # last position → next-event
        output: torch.Tensor = self.lm_head(h)     # (B,V)
        return output
