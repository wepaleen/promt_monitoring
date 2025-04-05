from datetime import datetime

# Датасет для тестового задания
PROMPT_TUNING_DATASET = {
    "name": "prompt_tuning_tutorial",
    "description": """
    Датасет для демонстрации возможностей тюнинга промптов с использованием Langfuse.
    Включает различные типы запросов для тестирования разных аспектов работы с LLM.
    """,
    "items": [
        # 1. Базовый запрос для проверки структуры ответа
        {
            "input": {
                "text": "Объясни, что такое prompt tuning и зачем он нужен?"
            },
            "expected_output": {
                "text": """
                Prompt tuning - это процесс оптимизации текстовых запросов к языковым моделям.

                Основные компоненты:
                - Структура промпта
                - Параметры запроса
                - Метрики качества
                - Системы мониторинга

                Преимущества:
                - Улучшение качества ответов
                - Снижение стоимости запросов
                - Повышение стабильности результатов

                Пример использования:
                ```python
                optimized_prompt = f'''
                Инструкция: {instruction}
                Контекст: {context}
                Запрос: {query}
                '''
                ```
                """
            },
            "metadata": {
                "type": "explanation",
                "complexity": "intermediate",
                "required_elements": ["definition", "components", "benefits", "example"]
            }
        },

        # 2. Запрос для проверки аналитических способностей
        {
            "input": {
                "text": "Сравни разные подходы к тюнингу промптов и их эффективность"
            },
            "expected_output": {
                "text": """
                Сравнение подходов к тюнингу промптов:

                1. Ручной тюнинг:
                ✓ Простота реализации
                ✗ Трудоемкость
                ✗ Отсутствие автоматизации

                2. Автоматизированный тюнинг (Langfuse):
                ✓ Системный подход
                ✓ Метрики и аналитика
                ✓ Масштабируемость
                ✗ Требует настройки

                3. Гибридный подход:
                ✓ Гибкость
                ✓ Баланс контроля и автоматизации
                ✗ Сложность координации

                Рекомендации:
                - Начинать с автоматизированного подхода
                - Использовать метрики для оценки
                - Регулярно обновлять промпты
                """
            },
            "metadata": {
                "type": "comparison",
                "complexity": "advanced",
                "required_elements": ["pros_cons", "recommendations", "structured_format"]
            }
        },

        # 3. Запрос для проверки технических деталей
        {
            "input": {
                "text": "Опиши процесс интеграции Langfuse в Python-приложение"
            },
            "expected_output": {
                "text": """
                Пошаговая инструкция по интеграции Langfuse:

                1. Установка:
                ```bash
                pip install langfuse openai python-dotenv
                ```

                2. Настройка окружения:
                ```python
                from langfuse import Langfuse
                from dotenv import load_dotenv
                import os

                load_dotenv()
                langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
                )
                ```

                3. Создание трейса:
                ```python
                with langfuse.trace(name="example") as trace:
                    # Ваш код здесь
                    pass
                ```

                4. Добавление метрик:
                ```python
                trace.update(
                    output=response,
                    metadata={"tokens": token_count}
                )
                ```
                """
            },
            "metadata": {
                "type": "technical",
                "complexity": "advanced",
                "required_elements": ["code_examples", "steps", "configuration"]
            }
        },

        # 4. Запрос для проверки метрик и аналитики
        {
            "input": {
                "text": "Какие метрики важно отслеживать при тюнинге промптов и почему?"
            },
            "expected_output": {
                "text": """
                Ключевые метрики для тюнинга промптов:

                1. Качественные метрики:
                - Точность ответов (accuracy)
                - Релевантность контента
                - Следование инструкциям
                
                2. Количественные метрики:
                - Использование токенов
                - Время ответа
                - Стоимость запросов

                3. Пользовательские метрики:
                - Удовлетворенность ответами
                - Количество повторных запросов
                - Процент успешных взаимодействий

                Пример анализа в Langfuse:
                ```python
                trace.score(
                    name="response_quality",
                    value=0.95,
                    comment="High accuracy and relevance"
                )
                ```

                Важность метрик:
                - Помогают оптимизировать затраты
                - Улучшают качество ответов
                - Выявляют проблемные области
                """
            },
            "metadata": {
                "type": "analytics",
                "complexity": "intermediate",
                "required_elements": ["metrics_list", "examples", "importance"]
            }
        },

        # 5. Запрос для проверки обработки ошибок
        {
            "input": {
                "text": "Как правильно обрабатывать ошибки и edge cases при работе с LLM?"
            },
            "expected_output": {
                "text": """
                Обработка ошибок при работе с LLM:

                1. Типичные ошибки:
                - Таймауты
                - Превышение токенов
                - Некорректные ответы
                
                2. Стратегии обработки:
                ```python
                try:
                    with langfuse.trace(name="llm_request") as trace:
                        response = llm.generate(prompt)
                        if not validate_response(response):
                            raise ValueError("Invalid response")
                        trace.update(output=response)
                except Exception as e:
                    trace.error(str(e))
                    fallback_response()
                ```

                3. Лучшие практики:
                - Валидация входных данных
                - Мониторинг ошибок
                - Fallback сценарии
                - Логирование проблем

                4. Превентивные меры:
                - Тестирование edge cases
                - Регулярный аудит ответов
                - Обновление промптов
                """
            },
            "metadata": {
                "type": "error_handling",
                "complexity": "advanced",
                "required_elements": ["error_types", "code_example", "best_practices"]
            }
        }
    ]
}

# Список всех доступных датасетов
AVAILABLE_DATASETS = {
    "prompt_tuning_tutorial": PROMPT_TUNING_DATASET
} 