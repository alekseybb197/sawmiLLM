# ADR-003: Формат данных — `event_id` последовательности + `section_bucket`

**Дата:** 2025-11-10  
**Статус:** Принято  
**Контекст:**  
Одинаковые события (например, `"ContinueOnError: False"`) могут встречаться в разных контекстах: "Prepare job", "Build Docker", "Deploy".  
Предсказание следующего события без учета контекста секции приведет к высокому уровню ложных срабатываний.

**Решение:**  
Добавить **section bucket ID** как отдельный embedding:
- каждая уникальная пара `pipeline_id + section_name` хешируется в bucket 0…2047;  
- embedding вектор суммируется с token и positional embeddings;  
- размер embedding-матрицы: `2048 × 512`.

Входные данные для модели `UnifiedLogBERT` состоят из двух компонентов:
- `input_ids`: последовательность `event_id` (из `drain_sequences.jsonl`)  
- `sec_ids`: `section_bucket` — целочисленный идентификатор секции (из `drain_mapping.jsonl`)

Эмбеддинг секции (`section_emb`) добавляется к каждому токену через бродкастинг:
```python
x = token_emb(input_ids) + section_emb(sec_ids).unsqueeze(1)
```

**Последствия:**  
✅ **Плюсы:**  
- Модель учитывает **контекст выполнения**  
- Поддержка сложных пайплайнов с повторяющимися шаблонами  
- Совместимость с `train_meta.jsonl` (включает `section_name`, `section_bucket`, `target_pos_in_section`)
- Снижение ложных срабатываний на «разрешенных» ошибках в конкретной секции.  
- Улучшение top-1 accuracy на 3–4 % на валидации.

⚠️ **Минусы:**  
- Увеличение числа параметров на ~1 M.  
- Необходимость сохранять маппинг `section_name → section_bucket` между train и inference.  
- Новые секции после деплоя попадают в «unknown» bucket (0).
- Ограничение на количество бакетов (`section_buckets=2048`)

**Связанные файлы:**  
- `drain_mapping.jsonl`  
- `train_meta.jsonl`  
- `event_vocab.json`

**Решение принято:** AB, 2025-11-10
