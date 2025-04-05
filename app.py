import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe
import logging
import time
import json
from datetime import datetime
import httpx
from datasets import AVAILABLE_DATASETS, PROMPT_TUNING_DATASET

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Инициализация Flask
app = Flask(__name__)

# Инициализация Langfuse
logger.info("Инициализация клиента Langfuse...")
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
)

# Проверка аутентификации
try:
    logger.info("Проверка аутентификации Langfuse...")
    langfuse.auth_check()
    logger.info("Langfuse аутентификация успешна")
except Exception as e:
    logger.error(f"Ошибка аутентификации Langfuse: {str(e)}")
    raise

# Создаем HTTP-клиент с совместимой версией httpx
http_client = httpx.Client()

# Инициализация OpenAI клиента
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

def create_dataset(name, description):
    """Создание датасета в Langfuse"""
    try:
        dataset = langfuse.create_dataset(
            name=name,
            description=description,
            metadata={
                "author": "prompt_tuning_system",
                "type": "benchmark",
                "created_at": datetime.now().isoformat()
            }
        )
        logger.info(f"Датасет {name} создан успешно")
        return dataset
    except Exception as e:
        logger.error(f"Ошибка при создании датасета: {str(e)}")
        raise

def add_dataset_item(dataset_name, input_data, expected_output, metadata=None):
    """Добавление элемента в датасет"""
    try:
        item = langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=input_data,
            expected_output=expected_output,
            metadata=metadata or {}
        )
        logger.info(f"Элемент добавлен в датасет {dataset_name}")
        return item
    except Exception as e:
        logger.error(f"Ошибка при добавлении элемента в датасет: {str(e)}")
        raise

def run_experiment(dataset_name, prompt_versions, model="gpt-3.5-turbo"):
    """Запуск эксперимента с разными версиями промптов"""
    try:
        dataset = langfuse.get_dataset(dataset_name)
        results = []
        
        for item in dataset.items:
            item_results = []
            
            for version, prompt_func in prompt_versions:
                # Создаем трейс
                trace = langfuse.trace(
                    name=f"experiment_{version}",
                    input=item.input,
                    metadata={
                        "dataset": dataset_name,
                        "version": version,
                        "model": model,
                        "item_id": item.id
                    }
                )
                
                try:
                    # Создаем промпт
                    prompt = prompt_func(item.input["text"])
                    
                    # Запускаем модель
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    
                    # Получаем ответ
                    answer = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens
                    
                    # Анализ ответа
                    word_count = len(answer.split())
                    has_examples = "пример" in answer.lower() or "example" in answer.lower()
                    has_practical_advice = "совет" in answer.lower() or "advice" in answer.lower()
                    has_code_blocks = "```" in answer
                    has_bullet_points = "- " in answer or "* " in answer
                    
                    # Оценка качества
                    quality_score = 0
                    if has_examples: quality_score += 1
                    if has_practical_advice: quality_score += 1
                    if has_code_blocks: quality_score += 1
                    if has_bullet_points: quality_score += 1
                    if 100 <= word_count <= 300: quality_score += 1
                    
                    # Сохраняем результаты
                    trace.update(
                        output={
                            "answer": answer,
                            "tokens_used": tokens_used,
                            "quality_score": quality_score
                        },
                        metadata={
                            "quality_metrics": {
                                "score": quality_score,
                                "max_score": 5,
                                "criteria": {
                                    "has_examples": has_examples,
                                    "has_practical_advice": has_practical_advice,
                                    "has_code_blocks": has_code_blocks,
                                    "has_bullet_points": has_bullet_points,
                                    "word_count_optimal": 100 <= word_count <= 300
                                }
                            }
                        }
                    )
                    
                    # Добавляем оценку
                    langfuse.score(
                        trace_id=trace.id,
                        name="quality",
                        value=quality_score,
                        comment=f"Оценка качества версии {version}"
                    )
                    
                    item_results.append({
                        "version": version,
                        "answer": answer,
                        "tokens_used": tokens_used,
                        "quality_score": quality_score
                    })
                except Exception as e:
                    logger.error(f"Ошибка при обработке версии {version}: {str(e)}")
                    trace.update(
                        output={"error": str(e)},
                        metadata={"error": True}
                    )
                    continue
            
            results.append({
                "input": item.input,
                "expected_output": item.expected_output,
                "results": item_results
            })
        
        return results
    except Exception as e:
        logger.error(f"Ошибка при запуске эксперимента: {str(e)}")
        raise

def create_prompt_v1(query):
    """Базовая версия промпта"""
    return f"""
    Пожалуйста, ответь на следующий вопрос максимально подробно и информативно.
    Вопрос: {query}
    """

def create_prompt_v2(query):
    """Оптимизированная версия промпта с контекстом"""
    return f"""
    Контекст: Ты - эксперт в области технологий.
    
    Правила:
    1. Начни с краткого ответа (1-2 предложения)
    2. Приведи 2-3 конкретных примера
    3. Заверши практическим советом
    
    Вопрос: {query}
    """

def create_prompt_v3(query):
    """Версия промпта с шаблоном и ограничениями"""
    return f"""
    Ты - эксперт в области технологий. Ответь на вопрос, следуя шаблону:
    
    [Краткий ответ]
    - Основная мысль
    - Ключевой момент
    
    [Примеры]
    - Пример 1
    - Пример 2
    
    [Практическое применение]
    - Совет 1
    - Совет 2
    
    Вопрос: {query}
    
    Ограничения:
    - Не используй технический жаргон
    - Держи ответ в пределах 200 слов
    - Фокусируйся на практической пользе
    """

def create_initial_dataset():
    """Создание начального датасета при запуске приложения"""
    try:
        # Проверяем, существует ли датасет
        try:
            dataset = langfuse.get_dataset(PROMPT_TUNING_DATASET["name"])
            logger.info(f"Датасет {PROMPT_TUNING_DATASET['name']} уже существует")
            return dataset
        except Exception:
            # Если датасет не существует, создаем его
            dataset = create_dataset(
                name=PROMPT_TUNING_DATASET["name"],
                description=PROMPT_TUNING_DATASET["description"]
            )
            
            # Добавляем элементы в датасет
            for item in PROMPT_TUNING_DATASET["items"]:
                add_dataset_item(
                    dataset_name=PROMPT_TUNING_DATASET["name"],
                    input_data=item["input"],
                    expected_output=item["expected_output"],
                    metadata=item["metadata"]
                )
            
            logger.info(f"Датасет {PROMPT_TUNING_DATASET['name']} создан успешно")
            return dataset
    except Exception as e:
        logger.error(f"Ошибка при создании начального датасета: {str(e)}")
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/datasets")
def get_datasets():
    """Получение списка доступных датасетов"""
    try:
        return jsonify({
            "status": "success",
            "datasets": list(AVAILABLE_DATASETS.keys())
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/dataset/<name>")
def get_dataset(name):
    """Получение информации о конкретном датасете"""
    try:
        if name not in AVAILABLE_DATASETS:
            return jsonify({
                "status": "error",
                "message": f"Dataset {name} not found"
            }), 404
        
        return jsonify({
            "status": "success",
            "dataset": AVAILABLE_DATASETS[name]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/create_dataset", methods=["POST"])
def create_dataset_route():
    """Создание нового датасета"""
    data = request.get_json()
    name = data.get("name")
    description = data.get("description")
    
    try:
        dataset = create_dataset(name, description)
        return jsonify({
            "status": "success",
            "dataset": {
                "name": dataset.name,
                "description": dataset.description,
                "metadata": dataset.metadata
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/add_item", methods=["POST"])
def add_item_route():
    """Добавление элемента в датасет"""
    data = request.get_json()
    dataset_name = data.get("dataset_name")
    input_data = data.get("input")
    expected_output = data.get("expected_output")
    metadata = data.get("metadata")
    
    try:
        item = add_dataset_item(dataset_name, input_data, expected_output, metadata)
        return jsonify({
            "status": "success",
            "item": {
                "id": item.id,
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/run_experiment", methods=["POST"])
@observe()
def run_experiment_route():
    """Запуск эксперимента"""
    data = request.get_json()
    dataset_name = data.get("dataset_name")
    model = data.get("model", "gpt-3.5-turbo")
    
    prompt_versions = [
        (1, create_prompt_v1),
        (2, create_prompt_v2),
        (3, create_prompt_v3)
    ]
    
    try:
        results = run_experiment(dataset_name, prompt_versions, model)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Создаем директорию для результатов, если её нет
    os.makedirs("results", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Создаем начальный датасет
    create_initial_dataset()
    
    app.run(host='0.0.0.0', port=5000, debug=True) 