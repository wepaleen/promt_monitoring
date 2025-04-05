# Langfuse Demo Python

Демонстрационный проект для тюнинга промптов с использованием Langfuse и OpenAI.


```
OPENAI_API_KEY=your_openai_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Запуск

```bash
python app.py
```

Сервер запустится на http://localhost:5001

## Использование

Отправьте POST запрос на эндпоинт `/ask`:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"query":"Расскажи, в чем разница между Python и Java?"}' \
    http://localhost:5001/ask
```

## Мониторинг

Все запросы будут логироваться в Langfuse. Вы можете:
- Отслеживать использование токенов
- Анализировать качество ответов
- Тюнинговать промпты
- Сравнивать разные версии промптов
