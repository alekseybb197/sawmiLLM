# ADR-001: Архитектура флоу получения логов из Azure DevOps

**Дата:** 2025-11-10  
**Статус:** Принято  
**Контекст:**  
Для обучения и инференса модели `UnifiedLogBERT` требуются **сырые логи** из CI/CD-пайплайнов Azure DevOps.  
Логи содержат динамические значения, временные метки, декораторы и другие элементы, мешающие кластеризации и обучению.  
Нужно реализовать **автоматизированный флоу** получения, нормализации и подготовки логов.

**Проблема:**  
- Azure DevOps API — сложная иерархия: `Organization` → `Project` → `Pipelines` → `Builds` → `Logs`
- Логи разбросаны по нескольким файлам в рамках одного билда
- Требуется аутентификация через PAT-токен
- Нужно хранить конфигурацию (URL, проект, токен) отдельно от кода

**Решение:**  
Реализовать **трёхэтапный флоу получения логов**:
1. **Получение списка пайплайнов** → `get-pipelines.py` → `pipelines.csv`
2. **Получение билдов для каждого пайплайна** → `get-builds.py` → `pipeline_id-successed-builds.csv`
3. **Получение логов для каждого билда** → `get-logs.py` → `pipeline_id-build_id-logs.csv`, `pipeline_id-build_id-logs_content.txt`

Конфигурация доступа к API хранится в `.env` и используется всеми скриптами.

**Детали архитектуры:**

- **`.env`**:
  - Содержит: `TFS_PAT_TOKEN`, `TFS_ACCOUNT`, `TFS_ORGANIZATION`, `TFS_PROJECT`, `TFS_API_URL`
  - Используется всеми Python-скриптами через `python-dotenv`
  - Не коммитится в репозиторий

- **`get-pipelines.py`**:
  - Выполняет: `GET /_apis/pipelines`
  - Вывод: `pipelines.csv` (ID, Name, Folder)

- **`get-builds.py {pipeline_id}`**:
  - Выполняет: `GET /_apis/pipelines/{id}/runs`
  - Фильтрует: только `completed` и `succeeded`
  - Вывод: `pipeline_id-successed-builds.csv` (Run ID, Name, State, Result, Dates)

- **`get-logs.py {pipeline_id} {build_id}`**:
  - Выполняет: `GET /_apis/pipelines/{id}/runs/{buildId}/logs`
  - Вывод: `pipeline_id-build_id-logs.csv` (log metadata) + `pipeline_id-build_id-logs_content.txt` (сырые логи)

**Последствия:**

✅ **Плюсы:**
- Чёткое разделение этапов: можно запускать по отдельности
- Масштабируемость: можно запускать параллельно для разных пайплайнов
- Прозрачность: каждый этап сохраняет промежуточные CSV-файлы
- Безопасность: токены не в коде, а в `.env`
- Воспроизводимость: логи хранятся в файловой системе

⚠️ **Минусы:**
- Требуется хранение промежуточных CSV-файлов (диск)
- Ограничения API: рейт-лимиты, размер ответа
- Зависимость от стабильности Azure DevOps API

**Связанные файлы:**
- `.env`
- `get_pipelines.py`
- `get_builds.py`
- `get_logs.py`
- `pipelines.csv`, `pipeline_id-successed-builds.csv`, `pipeline_id-build_id-logs.csv`, `pipeline_id-build_id-logs_content.txt`

**Пример использования:**
```bash
# 1. Загрузить пайплайны
python get_pipelines.py

# 2. Для каждого pipeline_id получить билды
python get_builds.py pipeline_id5

# 3. Для каждого билда получить логи
python get_logs.py pipeline_id5 build_id1
```

**Решение принято:** AB, 2025-11-10
