#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drain_stats.py — анализ текущего состояния Drain индекса (drain_state.json)

Пример использования:
    python 92-drain_stats.py                    # Использует dataset.drain из config.yaml
    python 92-drain_stats.py --state path/to/drain_state.json  # Явно указанный путь
"""

import argparse
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence

from modules.config import get_config
from modules.paths import get_dataset_path_from_config


def analyze_drain_state(state_path: Path):
    if not state_path.exists():
        print(f"❌ Файл не найден: {state_path}")
        return

    try:
        # Загружаем состояние Drain3 через FilePersistence
        persistence = FilePersistence(str(state_path))
        miner = TemplateMiner(persistence_handler=persistence)
        
        # Получаем кластеры из дерева Drain
        clusters = miner.drain.clusters
        n_clusters = len(clusters)
        
        if n_clusters == 0:
            print("⚠️ В Drain state нет кластеров (возможно файл пустой или повреждён).")
            return

        total_examples = 0
        token_lengths = []
        top_freq = []
        
        for cluster in clusters:
            # Получаем шаблон и размер кластера
            template = cluster.get_template()
            count = cluster.size
            
            total_examples += count
            
            # Обрабатываем шаблон (может быть строкой или списком токенов)
            if isinstance(template, list):
                template_str = " ".join(template)
                token_lengths.append(len(template))
            else:
                template_str = template
                token_lengths.append(len(template.split()) if template else 0)
            
            top_freq.append((count, cluster.cluster_id, template_str))

        avg_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        avg_count = total_examples / n_clusters if n_clusters else 0
        size_mb = state_path.stat().st_size / (1024 ** 2)

        print("📊 Drain State Summary")
        print("=" * 60)
        print(f"📁 Файл:           {state_path}")
        print(f"💾 Размер:         {size_mb:.2f} MB")
        print(f"🧩 Кластеров:      {n_clusters:,}")
        print(f"🧮 Всего примеров: {total_examples:,}")
        print(f"⚖️  Сред. по кластеру: {avg_count:.1f} логов")
        print(f"🧱 Сред. длина шаблона: {avg_len:.1f} токенов")

        print("\n🏆 Топ-10 самых частых шаблонов:")
        for count, cid, tmpl in sorted(top_freq, key=lambda x: x[0], reverse=True)[:10]:
            print(f"  • E{cid:>5} — {count:,}×  → {tmpl[:80]}{'...' if len(tmpl) > 80 else ''}")

        print("=" * 60)
        print("✅ Анализ завершён.")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки состояния Drain3: {e}")
        import traceback
        traceback.print_exc()
        return

def main():
    ap = argparse.ArgumentParser(
        description="Анализ состояния Drain индекса (drain_state.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                           # Использует dataset.drain/drain_state.json из config.yaml
  %(prog)s --state path/to/drain_state.json  # Явно указанный путь к файлу
        """
    )
    ap.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Путь к drain_state.json (по умолчанию: dataset.drain/drain_state.json из config.yaml)"
    )
    args = ap.parse_args()
    
    # Если путь не указан, используем путь из config.yaml
    if args.state is None:
        try:
            config = get_config()
            drain_dir = get_dataset_path_from_config(config, "drain")
            
            if drain_dir is None:
                print("❌ Ошибка: поле 'dataset.drain' не указано в config.yaml")
                print("   Укажите путь явно через --state или добавьте dataset.drain в config.yaml")
                return
            
            # Формируем путь к drain_state.json
            state_path = drain_dir / "drain_state.json"
            print(f"📁 Используется путь из config.yaml: {state_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            print("   Укажите путь явно через --state")
            return
    else:
        state_path = args.state
    
    analyze_drain_state(state_path)

if __name__ == "__main__":
    main()
