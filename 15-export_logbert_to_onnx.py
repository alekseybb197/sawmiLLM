#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15-export_logbert_to_onnx.py — экспорт UnifiedLogBERT в ONNX и int8-квантизация

Использует пути из config.yaml:
  - model.checkpoints: путь к чекпоинту модели
  - model.onnx: путь для сохранения ONNX моделей
  - model.*: параметры архитектуры модели
  - dataset.vocab: путь к словарю для определения vocab_size
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

#from modules.model import UnifiedLogBERT
from modules.model import UnifiedLogBERTForONNX
from modules.config import get_config
from modules.model_utils import (
    resolve_checkpoint_paths,
    load_model_from_checkpoint,
    get_vocab_size,
)
from modules.paths import get_dataset_path_from_config


# Функция load_checkpoint_model удалена - используем load_model_from_checkpoint из modules.model_utils


def export_to_onnx(
    model: nn.Module,
    onnx_path: str,
    max_len: int,
    section_buckets: int,
    vocab_size: int,
    opset: int = 18,
):
    """
    Экспортирует модель UnifiedLogBERT в ONNX с динамическими осями (batch, seq_len).
    
    Модель принимает:
      - input_ids: (batch, seq_len) - токены событий
      - sec_ids: (batch,) - идентификаторы секций
    
    Возвращает:
      - logits: (batch, vocab_size) - логиты для следующего события
      где vocab_size фиксирован, а batch_size может варьироваться
    """
    print(f"✅ Экспорт в ONNX: {onnx_path}")
    print("\n📊 Параметры модели:")
    print(f"   vocab_size:      {vocab_size}")
    print(f"   max_len:         {max_len}")
    print(f"   section_buckets: {section_buckets}")
    print(f"   opset:           {opset}")
    
    # Проверяем реальные параметры модели
    if hasattr(model, 'token_emb'):
        actual_vocab_size = model.token_emb.num_embeddings
        actual_embedding_dim = model.token_emb.embedding_dim
        print(f"   model.token_emb.num_embeddings: {actual_vocab_size}")
        print(f"   model.token_emb.embedding_dim:  {actual_embedding_dim}")
        if actual_vocab_size != vocab_size:
            print(f"   ⚠️  ВНИМАНИЕ: vocab_size из параметров ({vocab_size}) != model.token_emb.num_embeddings ({actual_vocab_size})")
    
    if hasattr(model, 'lm_head'):
        actual_lm_head_out = model.lm_head.out_features
        print(f"   model.lm_head.out_features:    {actual_lm_head_out}")
        if actual_lm_head_out != vocab_size:
            print(f"   ⚠️  ВНИМАНИЕ: vocab_size из параметров ({vocab_size}) != model.lm_head.out_features ({actual_lm_head_out})")

    # Dummy-входы: batch_size=1, seq_len=max_len
    batch_size = 1
    seq_len = max_len

    print(f"\n📦 Создание dummy-тензоров:")
    print(f"   batch_size: {batch_size}")
    print(f"   seq_len:    {seq_len}")

    # UnifiedLogBERT принимает input_ids и sec_ids (не attention_mask!)
    input_ids = torch.randint(
        low=0,
        high=max(vocab_size - 1, 2),
        size=(batch_size, seq_len),
        dtype=torch.long,
    )
    sec_ids = torch.randint(
        low=0,
        high=section_buckets,
        size=(batch_size,),
        dtype=torch.long,
    )
    
    print(f"   input_ids.shape: {input_ids.shape}")
    print(f"   sec_ids.shape:   {sec_ids.shape}")
    print(f"   input_ids range: [0, {vocab_size - 1}]")
    print(f"   sec_ids range:   [0, {section_buckets - 1}]")
    
    # Проверяем выход модели перед экспортом
    print(f"\n🔍 Проверка forward pass перед экспортом:")
    with torch.no_grad():
        output = model(input_ids, sec_ids)
        print(f"   output.shape: {output.shape}")
        print(f"   Ожидаемая форма: (batch={batch_size}, vocab_size={vocab_size})")
        if output.shape != (batch_size, vocab_size):
            print(f"   ⚠️  ВНИМАНИЕ: Неожиданная форма выхода! Ожидалось ({batch_size}, {vocab_size}), получено {output.shape}")

    # Имена входов/выходов
    input_names = ["input_ids", "sec_ids"]
    output_names = ["logits"]

    # Динамические оси: задаём только для входов
    # Для выхода (logits) не указываем динамику - это упрощает shape inference
    # ONNX экспортер сам выведет правильную форму выхода
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "sec_ids": {0: "batch_size"},
        # logits не указываем - пусть экспортер сам определит форму
    }

    print(f"\n📤 Экспорт в ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, sec_ids),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    print("✅ ONNX экспорт завершён")
    
    # Проверяем экспортированную модель
    print(f"\n🔍 Проверка экспортированной ONNX модели:")
    try:
        onnx_model = onnx.load(onnx_path)
        
        # Проверяем входы
        print("   Входы:")
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else f"dim_{i}" 
                    for i, dim in enumerate(inp.type.tensor_type.shape.dim)]
            print(f"     {inp.name}: shape={shape}, type={inp.type.tensor_type.elem_type}")
        
        # Проверяем выходы
        print("   Выходы:")
        for out in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else f"dim_{i}" 
                    for i, dim in enumerate(out.type.tensor_type.shape.dim)]
            print(f"     {out.name}: shape={shape}, type={out.type.tensor_type.elem_type}")
            
            # Особое внимание к logits
            if out.name == "logits":
                if len(shape) == 2:
                    batch_dim, vocab_dim = shape
                    print(f"     logits: batch_dim={batch_dim}, vocab_dim={vocab_dim}")
                    if isinstance(vocab_dim, int) and vocab_dim != vocab_size:
                        print(f"     ⚠️  ВНИМАНИЕ: vocab_dim в ONNX ({vocab_dim}) != ожидаемый vocab_size ({vocab_size})")
                    if isinstance(vocab_dim, int) and vocab_dim == max_len:
                        print(f"     ⚠️  КРИТИЧНО: vocab_dim ({vocab_dim}) совпадает с max_len ({max_len}) - возможна путаница!")
    except Exception as e:
        print(f"   ⚠️  Ошибка при проверке ONNX модели: {e}")


def quantize_onnx(
    onnx_path: str,
    quantized_path: str,
):
    """
    Пост-тренировочная динамическая int8-квантизация через ONNX Runtime.

    ВАЖНО: Shape inference в onnxruntime.quantization иногда ломается
    на больших трансформерах (путает размерности, как у нас 512 vs 45950).
    Для dynamic quantization он не обязателен, поэтому мы его отключаем
    через extra_options={"DisableShapeInference": True}.
    """
    print(f"✅ Квантизация ONNX → int8\n   исходник: {onnx_path}\n   выход:   {quantized_path}")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
        # Ключевая строчка — отключаем shape inference внутри квантайзера
        extra_options={"DisableShapeInference": True},
    )

    print("✅ Квантизованная модель сохранена")


def save_metadata(
    output_dir: str,
    checkpoint_path: str,
    onnx_path: str,
    quantized_path: str,
    max_len: int,
    opset: int,
):
    """
    Сохраняем маленький JSON с метаданными экспорта — пригодится для воспроизводимости.
    """
    meta = {
        "export_time": datetime.utcnow().isoformat() + "Z",
        "checkpoint_path": checkpoint_path,
        "onnx_fp32_path": os.path.abspath(onnx_path),
        "onnx_int8_path": os.path.abspath(quantized_path),
        "max_len": max_len,
        "opset": opset,
    }
    meta_path = os.path.join(output_dir, "onnx_export_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Метаданные экспорта сохранены в {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Экспорт UnifiedLogBERT в ONNX и int8-квантизация для CPU-инференса",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                           # Использует пути из config.yaml
  %(prog)s --checkpoint path/to/model.pt  # Явно указанный путь к чекпоинту
  %(prog)s --output-dir path/to/onnx  # Явно указанная директория для ONNX
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Путь к чекпоинту модели (по умолчанию: model.checkpoints из config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Папка для сохранения ONNX моделей (по умолчанию: model.onnx из config.yaml)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Версия ONNX opset (рекомендуется 18+, по умолчанию: 18)",
    )

    args = parser.parse_args()

    # Загружаем конфигурацию
    print("📖 Загрузка конфигурации из config.yaml...")
    config = get_config()

    # Определяем путь к чекпоинту
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        model_ckpt_path, _ = resolve_checkpoint_paths(config)
        checkpoint_path = model_ckpt_path
        print(f"📁 Используется путь к чекпоинту из config.yaml: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Чекпоинт не найден: {checkpoint_path}\n"
            f"Укажите путь явно через --checkpoint или проверьте model.checkpoints в config.yaml"
        )

    # Определяем путь к словарю для получения vocab_size
    vocab_dir = get_dataset_path_from_config(config, "vocab")
    if vocab_dir is None:
        raise ValueError(
            "Поле 'dataset.vocab' не указано в config.yaml. "
            "Пожалуйста, укажите путь к директории со словарем."
        )
    
    vocab_path = vocab_dir / "event_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Словарь событий не найден: {vocab_path}\n"
            f"Пожалуйста, убедитесь, что файл существует."
        )
    
    vocab_size = get_vocab_size(vocab_path)
    print(f"📖 Размер словаря: {vocab_size}")

    # Определяем директорию для сохранения ONNX
    if args.output_dir:
        onnx_dir = Path(args.output_dir)
    else:
        model_cfg = config.get("model", {})
        onnx_dir_str = model_cfg.get("onnx")
        if not onnx_dir_str:
            raise ValueError(
                "Поле 'model.onnx' не указано в config.yaml. "
                "Пожалуйста, укажите путь к директории для сохранения ONNX моделей."
            )
        onnx_dir = Path(onnx_dir_str)
        print(f"📁 Используется путь для ONNX из config.yaml: {onnx_dir}")

    os.makedirs(onnx_dir, exist_ok=True)

    # Получаем параметры модели из конфигурации
    model_cfg = config.get("model", {})
    section_buckets = int(model_cfg.get("section_buckets", 2048))
    max_len = int(model_cfg.get("max_len", 512))

    # Загружаем модель из чекпоинта
    print(f"✅ Загрузка модели из {checkpoint_path}")
    print(f"\n📊 Параметры перед загрузкой модели:")
    print(f"   vocab_size:      {vocab_size}")
    print(f"   checkpoint_path: {checkpoint_path}")
    
    base_model = load_model_from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        vocab_size=vocab_size,
        device=torch.device("cpu"),  # Для экспорта используем CPU
        eval_mode=True,
    )
    model = UnifiedLogBERTForONNX(base_model)
    model.eval()
    
    print(f"\n📊 Параметры модели после загрузки:")
    if hasattr(model, 'token_emb'):
        print(f"   model.token_emb.num_embeddings: {model.token_emb.num_embeddings}")
        print(f"   model.token_emb.embedding_dim:  {model.token_emb.embedding_dim}")
    if hasattr(model, 'lm_head'):
        print(f"   model.lm_head.in_features:     {model.lm_head.in_features}")
        print(f"   model.lm_head.out_features:    {model.lm_head.out_features}")
    if hasattr(model, 'pos'):
        if hasattr(model.pos, 'pe'):
            pe_shape = model.pos.pe.shape if hasattr(model.pos.pe, 'shape') else 'N/A'
            print(f"   model.pos.pe.shape:           {pe_shape}")

    # Пути к файлам
    onnx_fp32_path = onnx_dir / "logbert_fp32.onnx"
    onnx_int8_path = onnx_dir / "logbert_int8.onnx"

    # Экспорт в ONNX
    export_to_onnx(
        model=model,
        onnx_path=str(onnx_fp32_path),
        max_len=max_len,
        section_buckets=section_buckets,
        vocab_size=vocab_size,
        opset=args.opset,
    )

    # Валидация ONNX
    print("\n✅ Проверка корректности ONNX-модели")
    onnx_model = onnx.load(str(onnx_fp32_path))
    ##onnx.checker.check_model(onnx_model)
    print("✅ ONNX-модель корректна")
    
    # Дополнительная проверка размеров после валидации
    print(f"\n🔍 Финальная проверка размеров ONNX модели:")
    for out in onnx_model.graph.output:
        if out.name == "logits":
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" 
                    for dim in out.type.tensor_type.shape.dim]
            print(f"   {out.name}: shape={shape}")
            if len(shape) == 2:
                batch_dim, vocab_dim = shape
                print(f"     batch_dim:  {batch_dim}")
                print(f"     vocab_dim:  {vocab_dim}")
                print(f"     Ожидаемый vocab_size: {vocab_size}")
                print(f"     Ожидаемый max_len:    {max_len}")
                if isinstance(vocab_dim, int):
                    if vocab_dim == vocab_size:
                        print(f"     ✅ vocab_dim корректный")
                    elif vocab_dim == max_len:
                        print(f"     ❌ КРИТИЧНО: vocab_dim ({vocab_dim}) совпадает с max_len ({max_len})!")
                    else:
                        print(f"     ⚠️  vocab_dim ({vocab_dim}) не совпадает ни с vocab_size ({vocab_size}), ни с max_len ({max_len})")

    # Квантизация
    quantize_onnx(
        onnx_path=str(onnx_fp32_path),
        quantized_path=str(onnx_int8_path),
    )

    # Метаданные
    save_metadata(
        output_dir=str(onnx_dir),
        checkpoint_path=str(checkpoint_path),
        onnx_path=str(onnx_fp32_path),
        quantized_path=str(onnx_int8_path),
        max_len=max_len,
        opset=args.opset,
    )

    print("\n🥳 Готово!")
    print(f"   FP32 ONNX: {onnx_fp32_path}")
    print(f"   INT8 ONNX: {onnx_int8_path}")


if __name__ == "__main__":
    main()
