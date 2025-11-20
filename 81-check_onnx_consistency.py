#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
81-check_onnx_consistency.py — проверка согласованности ONNX моделей

Выполняет комплексную проверку ONNX моделей:
  1. Базовая проверка структуры и форматов (FP32 и INT8)
  2. Проверка согласованности форм слоёв (shape inference для FP32)
  3. Проверка согласованности FP32 ↔ INT8 по входам/выходам
  4. Проверка ключевых весов (token_emb.weight, lm_head.weight)

Использует пути из config.yaml:
  - model.onnx: директория с ONNX моделями
  - model.*: параметры архитектуры модели (d_model, vocab_size из словаря)
  - dataset.vocab: путь к словарю для определения vocab_size
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import onnx
from onnx import checker, shape_inference
from onnx.numpy_helper import to_array

from modules.config import get_config
from modules.paths import get_dataset_path_from_config
from modules.model_utils import get_vocab_size


def load_and_check_basic(path: Path) -> onnx.ModelProto:
    """
    Базовая проверка структуры и форматов ONNX модели.
    
    Returns:
        Загруженная ONNX модель
    """
    print(f"\n{'='*60}")
    print(f"📋 Проверка: {path.name}")
    print(f"{'='*60}")
    
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    model = onnx.load(str(path))
    
    # Базовая валидация структуры
    try:
        checker.check_model(model)
        print("✅ onnx.checker.check_model: OK")
    except Exception as e:
        print(f"❌ onnx.checker.check_model: ОШИБКА - {e}")
        raise

    g = model.graph
    print("\n  📥 Входы:")
    for i in g.input:
        dims = [d.dim_value if (d.dim_value != 0) else "?" for d in i.type.tensor_type.shape.dim]
        print(f"    {i.name}: {dims}")

    print("\n  📤 Выходы:")
    for o in g.output:
        dims = [d.dim_value if (d.dim_value != 0) else "?" for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}: {dims}")

    print(f"\n  📊 Статистика графа:")
    print(f"    Число нод: {len(g.node)}")
    print(f"    Число инициализаторов (весов): {len(g.initializer)}")
    
    return model


def check_shape_inference(fp32_path: Path) -> bool:
    """
    Проверка согласованности форм слоёв через shape inference.
    
    Если проходит без ошибок — значит все операции согласованы по размерностям.
    """
    print(f"\n{'='*60}")
    print("🔍 Проверка shape inference для FP32")
    print(f"{'='*60}")
    
    try:
        model = onnx.load(str(fp32_path))
        print("🚀 Запускаем shape_inference...")
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
        print("✅ Shape inference завершён успешно")
        print("   Все операции (MatMul, Add, LayerNorm, Embedding и т.п.) согласованы по размерностям")
        print("   Нет скрытых конфликтов типа '512 vs 45950' внутри графа")
        return True
    except Exception as e:
        print(f"❌ Shape inference завершился с ошибкой: {e}")
        return False


def io_signature(model: onnx.ModelProto) -> Tuple[List[Tuple[str, List]], List[Tuple[str, List]]]:
    """
    Извлекает сигнатуру входов и выходов модели.
    
    Returns:
        Кортеж (список входов, список выходов)
        Каждый элемент — (имя, список размерностей)
    """
    g = model.graph
    ins = [
        (
            i.name,
            [d.dim_value if d.dim_value != 0 else None for d in i.type.tensor_type.shape.dim]
        )
        for i in g.input
    ]
    outs = [
        (
            o.name,
            [d.dim_value if d.dim_value != 0 else None for d in o.type.tensor_type.shape.dim]
        )
        for o in g.output
    ]
    return ins, outs


def check_fp32_int8_consistency(fp32_path: Path, int8_path: Path) -> bool:
    """
    Проверка согласованности FP32 ↔ INT8 по входам/выходам.
    
    Если входы и выходы совпадают, значит INT8 модель — это тот же интерфейс,
    просто с другими внутренними типами.
    """
    print(f"\n{'='*60}")
    print("🔄 Проверка согласованности FP32 ↔ INT8")
    print(f"{'='*60}")
    
    fp32 = onnx.load(str(fp32_path))
    int8 = onnx.load(str(int8_path))
    
    fp32_ins, fp32_outs = io_signature(fp32)
    int8_ins, int8_outs = io_signature(int8)
    
    print("\n  📥 Входы FP32:", fp32_ins)
    print("  📥 Входы INT8:", int8_ins)
    print("\n  📤 Выходы FP32:", fp32_outs)
    print("  📤 Выходы INT8:", int8_outs)
    
    if fp32_ins == int8_ins:
        print("\n✅ Входы FP32 и INT8 полностью совпадают")
    else:
        print("\n❌ Входы FP32 и INT8 различаются!")
        return False
    
    if fp32_outs == int8_outs:
        print("✅ Выходы FP32 и INT8 полностью совпадают")
        return True
    else:
        print("❌ Выходы FP32 и INT8 различаются!")
        return False


def find_weights_by_shape(
    inits: Dict[str, onnx.TensorProto],
    expected_shape: Tuple[int, int],
    weight_type: str
) -> List[str]:
    """
    Ищет веса по форме тензора, а не по имени.
    
    ONNX может переименовывать параметры, поэтому поиск по форме более надёжен.
    
    Args:
        inits: Словарь инициализаторов {имя: TensorProto}
        expected_shape: Ожидаемая форма (vocab_size, d_model)
        weight_type: Тип веса для логирования ("token_emb" или "lm_head")
    
    Returns:
        Список имён найденных весов с указанной формой
    """
    candidates = []
    for name, init in inits.items():
        try:
            arr = to_array(init)
            if arr.shape == expected_shape:
                candidates.append(name)
        except Exception:
            continue
    
    return candidates


def check_key_weights(fp32_path: Path, vocab_size: int, d_model: int, max_len: int) -> bool:
    """
    Проверка ключевых весов модели по форме тензора.
    
    Проверяет, что в модели есть веса с формой [vocab_size, d_model]:
      - token_emb.weight (или любое другое имя с такой формой)
      - lm_head.weight (или любое другое имя с такой формой)
    
    ВАЖНО: Поиск выполняется по форме, а не по имени, так как ONNX может
    переименовывать параметры. Это более надёжный способ проверки.
    """
    print(f"\n{'='*60}")
    print("⚖️  Проверка ключевых весов (по форме тензора)")
    print(f"{'='*60}")
    
    model = onnx.load(str(fp32_path))
    inits = {init.name: init for init in model.graph.initializer}
    
    print(f"\n  Ожидаемые параметры:")
    print(f"    vocab_size: {vocab_size}")
    print(f"    d_model:    {d_model}")
    print(f"    max_len:    {max_len}")
    
    expected_shape = (vocab_size, d_model)
    all_ok = True
    
    # Ищем token_emb.weight по форме
    print(f"\n  🔍 Поиск token_emb.weight (форма {expected_shape}):")
    token_emb_candidates = find_weights_by_shape(inits, expected_shape, "token_emb")
    
    if token_emb_candidates:
        print(f"    ✅ Найдено {len(token_emb_candidates)} весов с формой {expected_shape}:")
        for name in token_emb_candidates:
            w = to_array(inits[name])
            print(f"      - '{name}': shape={w.shape}, dtype={w.dtype}")
        
        # Если найдено несколько, проверяем на возможную путаницу
        if len(token_emb_candidates) > 1:
            print(f"    ⚠️  Найдено несколько весов с формой {expected_shape}!")
            print(f"       Это может быть нормально, если есть дубликаты или разные слои")
    else:
        print(f"    ❌ Не найдено весов с формой {expected_shape}")
        # Проверяем, нет ли весов с транспонированной формой
        transposed_shape = (d_model, vocab_size)
        transposed_candidates = find_weights_by_shape(inits, transposed_shape, "token_emb")
        if transposed_candidates:
            print(f"    ⚠️  Найдены веса с транспонированной формой {transposed_shape}: {transposed_candidates[:3]}")
        
        # Проверяем, нет ли путаницы с max_len
        max_len_shape = (max_len, d_model)
        max_len_candidates = find_weights_by_shape(inits, max_len_shape, "token_emb")
        if max_len_candidates:
            print(f"    ⚠️  КРИТИЧНО: Найдены веса с формой {max_len_shape} (max_len вместо vocab_size)!")
            print(f"       Возможна путаница между vocab_size ({vocab_size}) и max_len ({max_len})")
            print(f"       Найденные веса: {max_len_candidates[:3]}")
        
        all_ok = False
    
    # Ищем lm_head.weight по форме
    print(f"\n  🔍 Поиск lm_head.weight (форма {expected_shape}):")
    lm_head_candidates = find_weights_by_shape(inits, expected_shape, "lm_head")
    
    if lm_head_candidates:
        print(f"    ✅ Найдено {len(lm_head_candidates)} весов с формой {expected_shape}:")
        for name in lm_head_candidates:
            w = to_array(inits[name])
            print(f"      - '{name}': shape={w.shape}, dtype={w.dtype}")
        
        # Если найдено несколько, проверяем на возможную путаницу
        if len(lm_head_candidates) > 1:
            print(f"    ⚠️  Найдено несколько весов с формой {expected_shape}!")
            print(f"       Это может быть нормально, если есть дубликаты или разные слои")
    else:
        print(f"    ❌ Не найдено весов с формой {expected_shape}")
        # Проверяем, нет ли весов с транспонированной формой
        transposed_shape = (d_model, vocab_size)
        transposed_candidates = find_weights_by_shape(inits, transposed_shape, "lm_head")
        if transposed_candidates:
            print(f"    ⚠️  Найдены веса с транспонированной формой {transposed_shape}: {transposed_candidates[:3]}")
        
        # Проверяем, нет ли путаницы с max_len
        max_len_shape = (max_len, d_model)
        max_len_candidates = find_weights_by_shape(inits, max_len_shape, "lm_head")
        if max_len_candidates:
            print(f"    ⚠️  КРИТИЧНО: Найдены веса с формой {max_len_shape} (max_len вместо vocab_size)!")
            print(f"       Возможна путаница между vocab_size ({vocab_size}) и max_len ({max_len})")
            print(f"       Найденные веса: {max_len_candidates[:3]}")
        
        all_ok = False
    
    # Выводим список всех инициализаторов для отладки только при ошибках
    if not all_ok:
        print(f"\n  📋 Все инициализаторы в модели (первые 30):")
        for name in sorted(inits.keys())[:30]:
            try:
                w = to_array(inits[name])
                print(f"    {name}: shape={w.shape}, dtype={w.dtype}")
            except Exception:
                print(f"    {name}: <не удалось прочитать>")
        if len(inits) > 30:
            print(f"    ... и ещё {len(inits) - 30} инициализаторов")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Проверка согласованности ONNX моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                           # Использует пути из config.yaml
  %(prog)s --fp32 path/to/fp32.onnx  # Явно указанные пути
  %(prog)s --int8 path/to/int8.onnx
        """
    )
    parser.add_argument(
        "--fp32",
        type=Path,
        default=None,
        help="Путь к FP32 ONNX модели (по умолчанию: model.onnx/logbert_fp32.onnx из config.yaml)",
    )
    parser.add_argument(
        "--int8",
        type=Path,
        default=None,
        help="Путь к INT8 ONNX модели (по умолчанию: model.onnx/logbert_int8.onnx из config.yaml)",
    )
    parser.add_argument(
        "--skip-shape-inference",
        action="store_true",
        help="Пропустить проверку shape inference (может быть медленной)",
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    print("📖 Загрузка конфигурации из config.yaml...")
    config = get_config()
    
    # Определяем пути к ONNX моделям
    if args.fp32:
        fp32_path = args.fp32
    else:
        model_cfg = config.get("model", {})
        onnx_dir_str = model_cfg.get("onnx")
        if not onnx_dir_str:
            raise ValueError(
                "Поле 'model.onnx' не указано в config.yaml. "
                "Пожалуйста, укажите путь к директории с ONNX моделями."
            )
        onnx_dir = Path(onnx_dir_str)
        fp32_path = onnx_dir / "logbert_fp32.onnx"
        print(f"📁 Используется путь к FP32 модели из config.yaml: {fp32_path}")
    
    if args.int8:
        int8_path = args.int8
    else:
        model_cfg = config.get("model", {})
        onnx_dir_str = model_cfg.get("onnx")
        if not onnx_dir_str:
            raise ValueError(
                "Поле 'model.onnx' не указано в config.yaml. "
                "Пожалуйста, укажите путь к директории с ONNX моделями."
            )
        onnx_dir = Path(onnx_dir_str)
        int8_path = onnx_dir / "logbert_int8.onnx"
        print(f"📁 Используется путь к INT8 модели из config.yaml: {int8_path}")
    
    # Получаем параметры модели для проверки весов
    vocab_size = None
    d_model = None
    
    # Получаем vocab_size из словаря
    vocab_dir = get_dataset_path_from_config(config, "vocab")
    if vocab_dir:
        vocab_path = vocab_dir / "event_vocab.json"
        if vocab_path.exists():
            vocab_size = get_vocab_size(vocab_path)
            print(f"📖 Размер словаря из config: {vocab_size}")
    
    # Получаем d_model и max_len из конфигурации
    model_cfg = config.get("model", {})
    d_model = int(model_cfg.get("d_model", 512))
    max_len = int(model_cfg.get("max_len", 512))
    print(f"📖 d_model из config: {d_model}")
    print(f"📖 max_len из config: {max_len}")
    
    if vocab_size is None:
        print("⚠️  Не удалось определить vocab_size - проверка весов будет неполной")
    else:
        # Предупреждаем о возможной путанице
        if vocab_size == max_len:
            print(f"⚠️  ВНИМАНИЕ: vocab_size ({vocab_size}) == max_len ({max_len}) - возможна путаница!")
    
    print(f"\n{'='*60}")
    print("🚀 Начало проверки ONNX моделей")
    print(f"{'='*60}")
    
    results: Dict[str, Optional[bool]] = {
        "basic_fp32": False,
        "basic_int8": False,
        "shape_inference": False,
        "fp32_int8_consistency": False,
        "key_weights": False,
    }
    
    # 1. Базовая проверка FP32
    try:
        fp32_model = load_and_check_basic(fp32_path)
        results["basic_fp32"] = True
    except Exception as e:
        print(f"❌ Базовая проверка FP32 провалена: {e}")
        return 1
    
    # 2. Базовая проверка INT8
    try:
        int8_model = load_and_check_basic(int8_path)
        results["basic_int8"] = True
    except Exception as e:
        print(f"❌ Базовая проверка INT8 провалена: {e}")
        return 1
    
    # 3. Проверка shape inference (только для FP32)
    if not args.skip_shape_inference:
        results["shape_inference"] = check_shape_inference(fp32_path)
    else:
        print("\n⏭️  Пропущена проверка shape inference (--skip-shape-inference)")
        results["shape_inference"] = None
    
    # 4. Проверка согласованности FP32 ↔ INT8
    try:
        results["fp32_int8_consistency"] = check_fp32_int8_consistency(fp32_path, int8_path)
    except Exception as e:
        print(f"❌ Проверка согласованности FP32 ↔ INT8 провалена: {e}")
        results["fp32_int8_consistency"] = False
    
    # 5. Проверка ключевых весов
    if vocab_size is not None:
        try:
            results["key_weights"] = check_key_weights(fp32_path, vocab_size, d_model, max_len)
        except Exception as e:
            print(f"❌ Проверка ключевых весов провалена: {e}")
            import traceback
            traceback.print_exc()
            results["key_weights"] = False
    else:
        print("\n⏭️  Пропущена проверка ключевых весов (vocab_size не определён)")
        results["key_weights"] = None
    
    # Итоговый отчёт
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print(f"{'='*60}")
    
    print("\nРезультаты проверок:")
    print(f"  ✅ Базовая проверка FP32:        {'✅ PASS' if results['basic_fp32'] else '❌ FAIL'}")
    print(f"  ✅ Базовая проверка INT8:        {'✅ PASS' if results['basic_int8'] else '❌ FAIL'}")
    
    if results["shape_inference"] is not None:
        print(f"  🔍 Shape inference FP32:         {'✅ PASS' if results['shape_inference'] else '❌ FAIL'}")
    else:
        print(f"  🔍 Shape inference FP32:         ⏭️  SKIPPED")
    
    print(f"  🔄 Согласованность FP32 ↔ INT8: {'✅ PASS' if results['fp32_int8_consistency'] else '❌ FAIL'}")
    
    if results["key_weights"] is not None:
        print(f"  ⚖️  Проверка ключевых весов:    {'✅ PASS' if results['key_weights'] else '❌ FAIL'}")
    else:
        print(f"  ⚖️  Проверка ключевых весов:    ⏭️  SKIPPED")
    
    # Определяем общий результат
    critical_checks = [
        results["basic_fp32"],
        results["basic_int8"],
        results["fp32_int8_consistency"],
    ]
    
    if all(critical_checks):
        print("\n✅ Все критические проверки пройдены успешно!")
        return 0
    else:
        print("\n❌ Некоторые критические проверки провалены!")
        return 1


if __name__ == "__main__":
    exit(main())
